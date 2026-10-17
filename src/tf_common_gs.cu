/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2011, 2012, 2014, 2015  Oliver Weihe (o.weihe@t-online.de)

mfaktc is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

mfaktc is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with mfaktc.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "gpusieve.h"
#include "tf_autotune.cuh"
#include "gs_timing.cuh"
#include "gs_workload_tune.cuh"

#include <cuda_runtime.h>

void gpusieve_graph_sieve_params(mystuff_t *mystuff, int sieve_size, unsigned int *bitarray, cudaKernelNodeParams *p);

#undef RAW_GPU_BENCH // FIXME

#ifdef TF_96BIT
#ifdef SHORTCUT_75BIT
extern "C" __host__ int tf_class_75_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_75_gs
#else
extern "C" __host__ int tf_class_95_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_95_gs
#endif
#endif
#ifdef TF_BARRETT
#ifdef TF_BARRETT_76BIT_GS
    extern "C" __host__ int tf_class_barrett76_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_barrett76_gs
#ifndef DEBUG_GPU_MATH
#define MFAKTC_GS_VARIANTS gs_variants_barrett76
#define MFAKTC_GS_FAMILY  "barrett76"
#endif
#elif defined TF_BARRETT_77BIT_GS
    extern "C" __host__ int tf_class_barrett77_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_barrett77_gs
#ifndef DEBUG_GPU_MATH
#define MFAKTC_GS_VARIANTS gs_variants_barrett77
#define MFAKTC_GS_FAMILY  "barrett77"
#endif
#elif defined TF_BARRETT_79BIT_GS
extern "C" __host__ int tf_class_barrett79_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_barrett79_gs
#ifndef DEBUG_GPU_MATH
#define MFAKTC_GS_VARIANTS gs_variants_barrett79
#define MFAKTC_GS_FAMILY  "barrett79"
#endif
#elif defined TF_BARRETT_87BIT_GS
extern "C" __host__ int tf_class_barrett87_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_barrett87_gs
#ifndef DEBUG_GPU_MATH
#define MFAKTC_GS_VARIANTS gs_variants_barrett87
#define MFAKTC_GS_FAMILY  "barrett87"
#endif
#elif defined TF_BARRETT_88BIT_GS
extern "C" __host__ int tf_class_barrett88_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_barrett88_gs
#ifndef DEBUG_GPU_MATH
#define MFAKTC_GS_VARIANTS gs_variants_barrett88
#define MFAKTC_GS_FAMILY  "barrett88"
#endif
#else
extern "C" __host__ int tf_class_barrett92_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_barrett92_gs
#ifndef DEBUG_GPU_MATH
#define MFAKTC_GS_VARIANTS gs_variants_barrett92
#define MFAKTC_GS_FAMILY  "barrett92"
#endif
#endif
#endif
{
    int i;
    timeval timer;
    int96 factor, k_base;
    int192 b_preinit;
    int shiftcount, ln2b, count = 0;
    int numblocks;
    unsigned long long k_remaining;
    char string[50];
    int shared_mem_required;
#ifndef MFAKTC_GS_VARIANTS
    int threads_per_block = THREADS_PER_BLOCK;
#endif
    int factorsfound = 0;
    int buf = 0, have_tf[2] = { 0, 0 };
    int full_batches = 0;
    cudaError_t cuda_ret;
    const unsigned long long int original_k_min = k_min;
    // Set after a candidate buffer overflow; from then on this wrapper keeps the
    // larger buffer.
    static int conservative_candidate_buffer = 0;

    // If we've never initialized the GPU sieving code, do so now.
    //  gpusieve_init (mystuff); // moved to main() function!

    // If we haven't initialized the GPU sieving code for this Mersenne exponent, do so now.
    gpusieve_init_exponent(mystuff);

    // Init the timer
    timer_init(&timer);

    // Pre-calculate some values

    shiftcount = 0;
    while ((1ULL << shiftcount) < (unsigned long long int)mystuff->exponent)
        shiftcount++;
    //  printf("\n\nshiftcount = %d\n",shiftcount);
    shiftcount -= 1;
    ln2b = 1;
    while (ln2b < 20 || ln2b < mystuff->bit_min) // how much preprocessing is possible
    {
        shiftcount--;
        ln2b <<= 1;
        if (mystuff->exponent & (1 << (shiftcount))) ln2b++;
    }
    //  printf("shiftcount = %d\n",shiftcount);
    //  printf("ln2b = %d\n",ln2b);
    b_preinit.d5 = 0;
    b_preinit.d4 = 0;
    b_preinit.d3 = 0;
    b_preinit.d2 = 0;
    b_preinit.d1 = 0;
    b_preinit.d0 = 0;
    if (ln2b < 32)
        b_preinit.d0 = 1 << ln2b;
    else if (ln2b < 64)
        b_preinit.d1 = 1 << (ln2b - 32);
    else if (ln2b < 96)
        b_preinit.d2 = 1 << (ln2b - 64);
    else if (ln2b < 128)
        b_preinit.d3 = 1 << (ln2b - 96);
    else if (ln2b < 160)
        b_preinit.d4 = 1 << (ln2b - 128);
    else
        b_preinit.d5 = 1 << (ln2b - 160); // b_preinit = 2^ln2b

restart_class:
    k_min        = original_k_min;
    count        = 0;
    buf          = 0;
    have_tf[0]   = have_tf[1] = 0;
    full_batches = 0;
    if (mystuff->gpu_sieve_pipeline && mystuff->gs_tf_stream) {
        cudaStreamSynchronize(mystuff->gs_sieve_stream);
        cudaStreamSynchronize(mystuff->gs_tf_stream);
        // Both bit arrays are idle, so the sieve stream may wait on these
        // from the first batch onward (needed for a uniform pipeline step).
        cudaEventRecord(mystuff->gs_tf_done[0], mystuff->gs_tf_stream);
        cudaEventRecord(mystuff->gs_tf_done[1], mystuff->gs_tf_stream);
        have_tf[0] = have_tf[1] = 1;
    }

    /* set result array to 0, RES[0] is the factor count and RES[31] flags a candidate buffer overflow */
    cuda_ret = cudaMemset(mystuff->d_RES, 0, 32 * sizeof(int));
    if (cuda_ret != cudaSuccess) {
        printf("ERROR: could not reset TF results: %s\n", cudaGetErrorString(cuda_ret));
        return RET_CUDA_ERROR;
    }

#ifdef DEBUG_GPU_MATH
    cudaMemset(mystuff->d_modbasecase_debug, 0, 32 * sizeof(int));
#endif

    // At verbosity 3, report how the class's GPU time splits between the sieve and TF kernels
    cuda_ret = gs_timing_begin_class(&gs_class_timing, mystuff->verbosity >= 3 && mystuff->mode == MODE_NORMAL);
    if (cuda_ret != cudaSuccess) {
        printf("ERROR: could not create GPU timing events: %s\n", cudaGetErrorString(cuda_ret));
        return RET_CUDA_ERROR;
    }

    // Calculate the initial bit-to-clear values for this class
    gpusieve_init_class(mystuff, k_min);
    cudaDeviceSynchronize();

    // Generously estimate the shared memory requirements for the TF kernel
#ifdef RAW_GPU_BENCH
    shared_mem_required = 100; // no sieving = 100%
#else
    if (mystuff->gpu_sieve_primes < 54)
        shared_mem_required = 100; // no sieving = 100%
    else if (mystuff->gpu_sieve_primes < 310)
        shared_mem_required = 50; // 54 primes expect 48.30%
    else if (mystuff->gpu_sieve_primes < 1846)
        shared_mem_required = 38; // 310 primes expect 35.50%
    else if (mystuff->gpu_sieve_primes < 21814)
        shared_mem_required = 30; // 1846 primes expect 28.10%
    else if (mystuff->gpu_sieve_primes < 67894)
        shared_mem_required = 24; // 21814 primes expect 21.93%
    else
        shared_mem_required = 22; // 67894 primes expect 19.94%
#endif
    {
        // k_deltas[] holds one unsigned short per candidate.  The old allocation of
        // sizeof(int) per estimated candidate is a 2x margin; start with 1.5x, which
        // can fit more blocks per SM, and fall back to 2x if a block overflows.
        // A block never has more candidates than bits, so cap the capacity there.
        const size_t estimated_candidates = ((size_t)mystuff->gpu_sieve_processing_size * shared_mem_required + 99) / 100;
        size_t candidate_capacity =
            conservative_candidate_buffer ? estimated_candidates * 2 : (estimated_candidates * 3 + 1) / 2;
        if (candidate_capacity > (size_t)mystuff->gpu_sieve_processing_size)
            candidate_capacity = (size_t)mystuff->gpu_sieve_processing_size;
        shared_mem_required = (int)(((candidate_capacity * sizeof(unsigned short)) + 255) & ~(size_t)255);
    }

    // FIXME: can't use all the shared memory for GPU sieve, lets keep 1kiB spare...
    if (mystuff->verbosity >= 3) printf("shared_mem_required = %d bytes\n", shared_mem_required + 1024);

    if ((shared_mem_required + 1024) > mystuff->max_shared_memory_per_block ||
        (shared_mem_required + 1024) > mystuff->max_shared_memory) {
        printf("ERROR: Not enough shared memory available!\n");
        printf("       Need approximately %d bytes per block\n", shared_mem_required + 1024);
        printf("       You can lower GPUSieveProcessSize or increase GPUSievePrimes to lower\n");
        printf("       the amount of shared memory needed\n");
        exit(1);
    }

#ifdef MFAKTC_GS_VARIANTS
    // Until a variant has been timed, use the one-candidate variant with the
    // most resident threads according to the CUDA occupancy calculator.  Cache
    // the result because this host wrapper runs once for every factoring
    // class, while its resource use is constant for a run.
    const int num_variants = (int)(sizeof(MFAKTC_GS_VARIANTS) / sizeof(MFAKTC_GS_VARIANTS[0]));
    static int cached_shared_mem_required = -1;
    static int occupancy_variant = 0;
    int variant;
    if (cached_shared_mem_required != shared_mem_required) {
        int best_resident_threads = 0;

        occupancy_variant = 0;
        for (int v = 0; v < num_variants; v++) {
            const gs_variant_t *gv = &MFAKTC_GS_VARIANTS[v];
            int active_blocks      = 0;

            if (gv->warp_local) continue; // no shared-memory candidate list

            // Opt in to the required dynamic shared-memory size before asking the
            // occupancy calculator about a launch that uses it.  Some variants
            // (e.g. two candidates per thread) cannot accept the same limit.
            if (!gs_candidate_smem_fits(mystuff, shared_mem_required)) continue;
            cuda_ret = cudaFuncSetAttribute(gv->kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_required);
            if (cuda_ret != cudaSuccess) continue;
            cuda_ret = cudaFuncSetAttribute(gv->kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
            if (cuda_ret != cudaSuccess && mystuff->verbosity >= 2)
                printf("WARNING: could not set the TF shared-memory carveout: %s\n", cudaGetErrorString(cuda_ret));

            if (gv->cands == 1 && gv->bit_max64 == 0 &&
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, gv->kernel, gv->threads, shared_mem_required) == cudaSuccess &&
                active_blocks * gv->threads > best_resident_threads) {
                best_resident_threads = active_blocks * gv->threads;
                occupancy_variant     = v;
            }
        }
        cached_shared_mem_required = shared_mem_required;
        if (mystuff->verbosity >= 3)
            printf("GPU TF launch: %d threads/block (%d resident threads/SM)\n", MFAKTC_GS_VARIANTS[occupancy_variant].threads,
                   best_resident_threads);
    }
    variant = occupancy_variant;

    // Variants compiled for another factor size can't be used for this bit level.
    int applicable[GS_MAX_VARIANTS];
    const int num_applicable = gs_applicable_variants(MFAKTC_GS_VARIANTS, num_variants, mystuff->bit_min - 63, applicable);

    // Self-tests cycle through the variants so that each one is checked against known factors.
    static unsigned int selftest_variant = 0;
    if (mystuff->mode != MODE_NORMAL) variant = applicable[selftest_variant++ % num_applicable];

    static int tuned_device = -1, tuned_bit_min = -1, tuned_cat = -1;
    static unsigned int tuned_exponent = 0;
    static int tuned_variant = 0;
    static cudaGraphExec_t gs_batch_graph = NULL;
    static cudaGraphNode_t gs_graph_sieve_node, gs_graph_tf_node;
    static int gs_graph_nblocks = -1, gs_graph_smem = -1, gs_graph_pipeline = -1;
    static gs_kernel_t gs_graph_kernel = NULL;
    static int gs_graph_disabled = 0;
    int need_search = 0, current_device = -1;

    auto configure_smem_for = [&](int v) -> cudaError_t {
        const gs_variant_t *gv = &MFAKTC_GS_VARIANTS[v];

        if (gv->warp_local) return cudaSuccess;
        if (!gs_candidate_smem_fits(mystuff, shared_mem_required)) return cudaErrorInvalidValue;
        cudaError_t err = cudaFuncSetAttribute(gv->kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_required);
        if (err == cudaSuccess) cudaFuncSetAttribute(gv->kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        return err;
    };

    auto apply_tune_cfg = [&](const gs_tune_cfg_t *cfg, int v) -> cudaError_t {
        if (cfg->primes != mystuff->gpu_sieve_primes) {
            mystuff->gpu_sieve_primes = cfg->primes;
            gpusieve_rebuild_primes(mystuff);
            gpusieve_init_exponent(mystuff);
            gpusieve_init_class(mystuff, original_k_min);
            cudaDeviceSynchronize();
        }
        mystuff->gpu_sieve_processing_size = cfg->process_size;
        mystuff->gpu_sieve_segment_size    = cfg->segment_size;
        mystuff->gpu_sieve_atomic_marks    = cfg->atomic_marks;
        mystuff->gpu_sieve_batch_size      = cfg->batch_size;
        if (mystuff->gpu_sieve_batch_size > mystuff->gpu_sieve_size)
            mystuff->gpu_sieve_batch_size = mystuff->gpu_sieve_size;
        if (mystuff->gpu_sieve_processing_size > 0 && mystuff->gpu_sieve_batch_size % mystuff->gpu_sieve_processing_size != 0)
            mystuff->gpu_sieve_batch_size = mystuff->gpu_sieve_size;
        shared_mem_required = gs_candidate_shared_mem(mystuff, conservative_candidate_buffer);
        variant             = v;
        return configure_smem_for(v);
    };

    if (mystuff->mode == MODE_NORMAL && mystuff->gpu_sieve_autotune) {
        cuda_ret = cudaGetDevice(&current_device);
        if (cuda_ret != cudaSuccess) {
            printf("ERROR: cudaGetDevice() failed: %s\n", cudaGetErrorString(cuda_ret));
            return RET_CUDA_ERROR;
        }
        if (tuned_device == current_device && tuned_bit_min == mystuff->bit_min && tuned_cat == gs_exp_category(mystuff->exponent) &&
            tuned_exponent == mystuff->exponent) {
            variant = tuned_variant;
        } else {
            gs_tune_cfg_t cached;
            int v = -1;
            if (gs_tune_cache_load(MFAKTC_GS_FAMILY, mystuff->bit_min, mystuff->exponent, &cached))
                v = gs_tune_find_variant(MFAKTC_GS_VARIANTS, num_variants, &cached);
            if (v >= 0 && (MFAKTC_GS_VARIANTS[v].bit_max64 == 0 || MFAKTC_GS_VARIANTS[v].bit_max64 == mystuff->bit_min - 63)) {
                cuda_ret = apply_tune_cfg(&cached, v);
                if (cuda_ret != cudaSuccess) {
                    if (mystuff->verbosity >= 1)
                        printf("WARNING: could not apply cached GPU-sieve config (%s); re-autotuning\n",
                               cudaGetErrorString(cuda_ret));
                    cudaGetLastError();
                    cuda_ret    = cudaSuccess;
                    need_search = 1;
                } else {
                tuned_variant  = v;
                tuned_device   = current_device;
                tuned_bit_min  = mystuff->bit_min;
                tuned_cat      = gs_exp_category(mystuff->exponent);
                tuned_exponent = mystuff->exponent;
                if (mystuff->verbosity >= 2)
                    printf("GPU-sieve autotune cache: %d primes, process %d Kib, segment %d Kib, %s marks, batch %d Kib, "
                           "%d threads, %d candidate%s/thread, %s list\n",
                           cached.primes, cached.process_size / 1024, cached.segment_size / 1024,
                           cached.atomic_marks ? "atomic" : "sloppy", cached.batch_size / 1024, cached.tf_threads, cached.tf_cands,
                           cached.tf_cands == 1 ? "" : "s", cached.tf_warp_local ? "warp-local" : "block");
                }
            } else
                need_search = 1;
        }
    }
#else
    // Choose between 128- and 256-thread launches using the CUDA 13 occupancy
    // calculator.  Cache the result because this host wrapper runs once for
    // every factoring class, while its resource use is constant for a run.
    static int cached_shared_mem_required = -1;
    static int cached_threads_per_block = THREADS_PER_BLOCK;
    int active_blocks_128 = 0, active_blocks_256 = 0;
    if (cached_shared_mem_required != shared_mem_required) {
        // Opt in to the required dynamic shared-memory size before asking the
        // occupancy calculator about a launch that uses it.
        cuda_ret = cudaFuncSetAttribute(MFAKTC_FUNC, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_required);
        if (cuda_ret != cudaSuccess) {
            printf("ERROR: could not configure %d bytes of dynamic shared memory: %s\n", shared_mem_required,
                   cudaGetErrorString(cuda_ret));
            return RET_CUDA_ERROR;
        }
        cuda_ret = cudaFuncSetAttribute(MFAKTC_FUNC, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        if (cuda_ret != cudaSuccess && mystuff->verbosity >= 2)
            printf("WARNING: could not set the TF shared-memory carveout: %s\n", cudaGetErrorString(cuda_ret));

        cached_threads_per_block = THREADS_PER_BLOCK;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks_128, MFAKTC_FUNC, 128, shared_mem_required) == cudaSuccess &&
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks_256, MFAKTC_FUNC, THREADS_PER_BLOCK, shared_mem_required) ==
                cudaSuccess &&
            active_blocks_128 * 128 > active_blocks_256 * THREADS_PER_BLOCK) {
            cached_threads_per_block = 128;
        }
        cached_shared_mem_required = shared_mem_required;
    }
    threads_per_block = cached_threads_per_block;
    if (mystuff->verbosity >= 3 && active_blocks_128 != 0)
        printf("GPU TF launch: %d threads/block (%d vs %d resident threads/SM)\n", threads_per_block, active_blocks_128 * 128,
               active_blocks_256 * THREADS_PER_BLOCK);
#endif

    // Loop until all the k's are processed.  buf selects which of the two sieve
    // bit arrays the current batch uses when GPUSievePipeline is enabled.
    for (;;) {
        // Calculate the number of k's remaining.  Round this up so that we sieve an array that is
        // a multiple of the bits processed by each TF kernel (my_stuff->gpu_sieve_processing_size).

        k_remaining = ((k_max - k_min + 1) + NUM_CLASSES - 1) / NUM_CLASSES;
        if (k_remaining < (unsigned long long)mystuff->gpu_sieve_batch_size) {
            numblocks   = ((int)k_remaining + mystuff->gpu_sieve_processing_size - 1) / mystuff->gpu_sieve_processing_size;
            k_remaining = numblocks * mystuff->gpu_sieve_processing_size;
        } else
            numblocks = mystuff->gpu_sieve_batch_size / mystuff->gpu_sieve_processing_size;

        // Set the k value corresponding to the first bit in the bit array

        k_base.d0 = (int)(k_min & 0xFFFFFFFF);
        k_base.d1 = (int)(k_min >> 32);
        k_base.d2 = 0;

#ifdef MFAKTC_GS_VARIANTS
        // GPUSieveAutotune=0 still times TF kernel variants on the first full
        // sieve; =1 times sieve-plus-TF configurations of primes, process size,
        // segment geometry, batch size and the TF variant, then caches the winner.
        const int is_full_sieve = (k_min == original_k_min) &&
                                  (((k_max - k_min + 1) + NUM_CLASSES - 1) / NUM_CLASSES >= (unsigned long long)mystuff->gpu_sieve_size);

        if (mystuff->mode == MODE_NORMAL && need_search && is_full_sieve) {
            gs_tune_cfg_t best;
            unsigned int *scratch = NULL;
            cudaEvent_t ev0 = NULL, ev1 = NULL;
            float elapsed, best_ms = 1.0e30f, sieve_ms = 0.0f;
            int tried_primes[8], ntried_primes = 0;
            const int save_primes   = mystuff->gpu_sieve_primes;
            const int save_process  = mystuff->gpu_sieve_processing_size;
            const int save_segment  = mystuff->gpu_sieve_segment_size;
            const int save_atomic   = mystuff->gpu_sieve_atomic_marks;
            const int save_batch    = mystuff->gpu_sieve_batch_size;
            const int save_smem     = shared_mem_required;

            best.primes        = save_primes;
            best.process_size  = save_process;
            best.segment_size  = save_segment;
            best.atomic_marks  = save_atomic;
            best.batch_size    = mystuff->gpu_sieve_size;
            gs_tune_from_variant(&best, &MFAKTC_GS_VARIANTS[variant]);

            auto time_sieve = [&](unsigned long long bits, float *ms) -> cudaError_t {
                cudaError_t err = cudaEventRecord(ev0, 0);
                if (err == cudaSuccess) {
                    gpusieve(mystuff, bits);
                    err = cudaEventRecord(ev1, 0);
                }
                if (err == cudaSuccess) err = cudaEventSynchronize(ev1);
                if (err == cudaSuccess) err = cudaEventElapsedTime(ms, ev0, ev1);
                return err;
            };
            auto time_tf = [&](int v, int nblocks, int process_bits, int smem, float *ms) -> cudaError_t {
                const gs_variant_t *gv = &MFAKTC_GS_VARIANTS[v];
                const int saved_smem   = shared_mem_required;
                cudaError_t err;

                if (!gs_candidate_smem_fits(mystuff, smem)) return cudaErrorInvalidValue;
                shared_mem_required = smem;
                err                 = configure_smem_for(v);
                shared_mem_required = saved_smem;
                if (err != cudaSuccess) return err;

                err = cudaMemsetAsync(scratch, 0, 32 * sizeof(unsigned int), 0);
                if (err == cudaSuccess) err = cudaEventRecord(ev0, 0);
                if (err == cudaSuccess)
                    err = gs_launch_variant(gv, nblocks, gs_variant_shared_mem(gv, smem), 0, mystuff->exponent, k_base,
                                            mystuff->d_bitarray[0], (unsigned int)process_bits, shiftcount, b_preinit, scratch,
                                            mystuff->bit_min - 63);
                if (err == cudaSuccess) err = cudaEventRecord(ev1, 0);
                if (err == cudaSuccess) err = cudaEventSynchronize(ev1);
                if (err == cudaSuccess) err = cudaEventElapsedTime(ms, ev0, ev1);
                return err;
            };

            cuda_ret = cudaMalloc((void **)&scratch, 32 * sizeof(unsigned int));
            if (cuda_ret == cudaSuccess) cuda_ret = cudaEventCreate(&ev0);
            if (cuda_ret == cudaSuccess) cuda_ret = cudaEventCreate(&ev1);
            if (cuda_ret == cudaSuccess) cuda_ret = time_sieve(mystuff->gpu_sieve_size, &sieve_ms); // warmup
            if (cuda_ret == cudaSuccess) cuda_ret = time_sieve(mystuff->gpu_sieve_size, &sieve_ms);

            int proc_cands[4], nproc = 0;
            for (int kib = 8; kib <= 32; kib += 8)
                if (mystuff->gpu_sieve_size % (kib * 1024) == 0) proc_cands[nproc++] = kib * 1024;

            for (int p = 0; p < nproc && cuda_ret == cudaSuccess; p++) {
                const int process_bits = proc_cands[p];
                const int nblocks      = mystuff->gpu_sieve_size / process_bits;
                int prev_process       = mystuff->gpu_sieve_processing_size;
                int smem;

                mystuff->gpu_sieve_processing_size = process_bits;
                smem                               = gs_candidate_shared_mem(mystuff, conservative_candidate_buffer);
                mystuff->gpu_sieve_processing_size = prev_process;
                if (!gs_candidate_smem_fits(mystuff, smem)) continue;

                int tune_variants[GS_MAX_VARIANTS], ntune = 0;
                {
                    const int saved_smem = shared_mem_required;

                    shared_mem_required = smem;
                    for (int a = 0; a < num_applicable; a++) {
                        const int vi = applicable[a];
                        const gs_variant_t *gv = &MFAKTC_GS_VARIANTS[vi];

                        if (gv->warp_local || configure_smem_for(vi) == cudaSuccess) tune_variants[ntune++] = vi;
                    }
                    shared_mem_required = saved_smem;
                }
                if (ntune == 0) continue;

                auto launch = [&](int a, unsigned int *result) -> cudaError_t {
                    const int vi           = tune_variants[a];
                    const gs_variant_t *gv = &MFAKTC_GS_VARIANTS[vi];
                    const int saved        = shared_mem_required;
                    cudaError_t err;

                    shared_mem_required = smem;
                    err                 = configure_smem_for(vi);
                    shared_mem_required = saved;
                    if (err != cudaSuccess) return err;
                    return gs_launch_variant(gv, nblocks, gs_variant_shared_mem(gv, smem), 0, mystuff->exponent, k_base,
                                             mystuff->d_bitarray[0], (unsigned int)process_bits, shiftcount, b_preinit, result,
                                             mystuff->bit_min - 63);
                };
                int fastest = 0;
                cuda_ret    = mfaktc_time_variants(ntune, launch, &fastest);
                if (cuda_ret != cudaSuccess) break;
                cuda_ret = time_tf(tune_variants[fastest], nblocks, process_bits, smem, &elapsed);
                if (cuda_ret != cudaSuccess) break;
                elapsed += sieve_ms;
                if (elapsed < best_ms) {
                    best_ms           = elapsed;
                    best.process_size = process_bits;
                    gs_tune_from_variant(&best, &MFAKTC_GS_VARIANTS[tune_variants[fastest]]);
                    variant = tune_variants[fastest];
                }
            }
            mystuff->gpu_sieve_processing_size = best.process_size;

            int seg_cands[] = { 64 * 1024, 128 * 1024, 256 * 1024 };
            for (int s = 0; s < 3 && cuda_ret == cudaSuccess; s++) {
                for (int at = 0; at < 2 && cuda_ret == cudaSuccess; at++) {
                    if (seg_cands[s] == save_segment && at == save_atomic) continue;
                    mystuff->gpu_sieve_segment_size = seg_cands[s];
                    mystuff->gpu_sieve_atomic_marks = at;
                    float s_ms = 0.0f, t_ms = 0.0f;
                    cuda_ret = time_sieve(mystuff->gpu_sieve_size, &s_ms);
                    if (cuda_ret == cudaSuccess)
                        cuda_ret = time_tf(variant, mystuff->gpu_sieve_size / best.process_size, best.process_size,
                                           gs_candidate_shared_mem(mystuff, conservative_candidate_buffer), &t_ms);
                    if (cuda_ret == cudaSuccess && s_ms + t_ms < best_ms) {
                        best_ms           = s_ms + t_ms;
                        best.segment_size = seg_cands[s];
                        best.atomic_marks = at;
                    }
                }
            }
            mystuff->gpu_sieve_segment_size = best.segment_size;
            mystuff->gpu_sieve_atomic_marks = best.atomic_marks;

            int prime_cands[4] = { mystuff->gpu_sieve_primes_configured, 67894, 82486, 94000 };
            tried_primes[ntried_primes++] = save_primes;
            for (int pc = 0; pc < 4 && cuda_ret == cudaSuccess; pc++) {
                int req = prime_cands[pc];
                if (req < GPU_SIEVE_PRIMES_MIN) req = GPU_SIEVE_PRIMES_MIN;
                if (req > GPU_SIEVE_PRIMES_MAX) req = GPU_SIEVE_PRIMES_MAX;
                mystuff->gpu_sieve_primes = req;
                gpusieve_rebuild_primes(mystuff);
                int rounded = mystuff->gpu_sieve_primes, seen = 0;
                for (int t = 0; t < ntried_primes; t++)
                    if (tried_primes[t] == rounded) seen = 1;
                if (seen) continue;
                if (ntried_primes < 8) tried_primes[ntried_primes++] = rounded;
                gpusieve_init_exponent(mystuff);
                gpusieve_init_class(mystuff, original_k_min);
                cudaDeviceSynchronize();
                float s_ms = 0.0f, t_ms = 0.0f;
                int smem = gs_candidate_shared_mem(mystuff, conservative_candidate_buffer);
                if (!gs_candidate_smem_fits(mystuff, smem)) continue;
                if (cuda_ret == cudaSuccess) cuda_ret = time_sieve(mystuff->gpu_sieve_size, &s_ms);
                if (cuda_ret == cudaSuccess)
                    cuda_ret = time_tf(variant, mystuff->gpu_sieve_size / best.process_size, best.process_size, smem, &t_ms);
                if (cuda_ret == cudaSuccess && s_ms + t_ms < best_ms) {
                    best_ms     = s_ms + t_ms;
                    best.primes = rounded;
                }
            }

            mystuff->gpu_sieve_primes = best.primes;
            gpusieve_rebuild_primes(mystuff);
            gpusieve_init_exponent(mystuff);
            gpusieve_init_class(mystuff, original_k_min);
            cudaDeviceSynchronize();
            mystuff->gpu_sieve_segment_size = best.segment_size;
            mystuff->gpu_sieve_atomic_marks = best.atomic_marks;

            for (int div = 2; div <= 4 && cuda_ret == cudaSuccess; div *= 2) {
                int bits = mystuff->gpu_sieve_size / div;
                if (bits < mystuff->gpu_sieve_processing_size || bits % best.process_size != 0 || bits % best.segment_size != 0)
                    continue;
                float s_ms = 0.0f, t_ms = 0.0f;
                cuda_ret = time_sieve((unsigned long long)bits, &s_ms);
                if (cuda_ret == cudaSuccess)
                    cuda_ret = time_tf(variant, bits / best.process_size, best.process_size,
                                       gs_candidate_shared_mem(mystuff, conservative_candidate_buffer), &t_ms);
                if (cuda_ret == cudaSuccess && (s_ms + t_ms) * (float)div < best_ms) {
                    best_ms         = (s_ms + t_ms) * (float)div;
                    best.batch_size = bits;
                }
            }

            if (cuda_ret == cudaSuccess) {
                int v = gs_tune_find_variant(MFAKTC_GS_VARIANTS, num_variants, &best);
                if (v < 0) v = variant;
                mystuff->gpu_sieve_primes          = save_primes;
                mystuff->gpu_sieve_processing_size = save_process;
                mystuff->gpu_sieve_batch_size      = save_batch;
                shared_mem_required                = save_smem;
                cuda_ret                           = apply_tune_cfg(&best, v);
                variant                            = v;
            }
            if (ev1) cudaEventDestroy(ev1);
            if (ev0) cudaEventDestroy(ev0);
            if (scratch) cudaFree(scratch);
            if (cuda_ret != cudaSuccess) {
                const cudaError_t tune_err = cuda_ret;

                mystuff->gpu_sieve_primes          = save_primes;
                mystuff->gpu_sieve_processing_size = save_process;
                mystuff->gpu_sieve_segment_size    = save_segment;
                mystuff->gpu_sieve_atomic_marks    = save_atomic;
                mystuff->gpu_sieve_batch_size      = save_batch;
                shared_mem_required                = save_smem;
                gpusieve_rebuild_primes(mystuff);
                gpusieve_init_exponent(mystuff);
                gpusieve_init_class(mystuff, original_k_min);
                cudaDeviceSynchronize();
                variant = occupancy_variant;
                configure_smem_for(variant);
                if (mystuff->verbosity >= 1)
                    printf("WARNING: GPU-sieve autotune failed (%s); using INI settings\n", cudaGetErrorString(tune_err));
                cudaGetLastError();
                cuda_ret = cudaSuccess;
            } else {
                tuned_variant  = variant;
                tuned_device   = current_device;
                tuned_bit_min  = mystuff->bit_min;
                tuned_cat      = gs_exp_category(mystuff->exponent);
                tuned_exponent = mystuff->exponent;
                gs_tune_from_variant(&best, &MFAKTC_GS_VARIANTS[variant]);
                gs_tune_cache_save(MFAKTC_GS_FAMILY, mystuff->bit_min, mystuff->exponent, &best);
                if (mystuff->verbosity >= 1)
                    printf("GPU-sieve autotune: %d primes, process %d Kib, segment %d Kib, %s marks, batch %d Kib, "
                           "%d threads, %d candidate%s/thread, %s list (%.1f ms / %d Mib)\n",
                           best.primes, best.process_size / 1024, best.segment_size / 1024, best.atomic_marks ? "atomic" : "sloppy",
                           best.batch_size / 1024, best.tf_threads, best.tf_cands, best.tf_cands == 1 ? "" : "s",
                           best.tf_warp_local ? "warp-local" : "block", best_ms, mystuff->gpu_sieve_size / (1024 * 1024));
            }
            need_search = 0;
            k_remaining = ((k_max - k_min + 1) + NUM_CLASSES - 1) / NUM_CLASSES;
            if (k_remaining < (unsigned long long)mystuff->gpu_sieve_batch_size) {
                numblocks   = ((int)k_remaining + mystuff->gpu_sieve_processing_size - 1) / mystuff->gpu_sieve_processing_size;
                k_remaining = numblocks * mystuff->gpu_sieve_processing_size;
            } else
                numblocks = mystuff->gpu_sieve_batch_size / mystuff->gpu_sieve_processing_size;
        }
#endif

        // Do some sieving on the GPU.  With GPUSievePipeline, this writes bit
        // array `buf` on the sieve stream; trial factoring of the other array
        // can still be running.

        const int pipeline = mystuff->gpu_sieve_pipeline && mystuff->d_bitarray[1] != NULL;
        int launched_graph = 0;
        cudaStream_t sieve_stream = pipeline ? mystuff->gs_sieve_stream : 0;
        cudaStream_t tf_stream    = pipeline ? mystuff->gs_tf_stream : 0;
        if (pipeline && k_min == original_k_min) {
            cuda_ret = cudaStreamSynchronize(0); // drain autotune / init_class on the default stream
            if (cuda_ret != cudaSuccess) {
                printf("ERROR: GPU-sieve pipeline setup failed: %s\n", cudaGetErrorString(cuda_ret));
                return RET_CUDA_ERROR;
            }
        }
#ifdef MFAKTC_GS_VARIANTS
        {
            const int full_batch = numblocks == mystuff->gpu_sieve_batch_size / mystuff->gpu_sieve_processing_size;
            if (full_batch) full_batches++;
            if (mystuff->gpu_sieve_graphs && !gs_graph_disabled && mystuff->mode == MODE_NORMAL && !gs_class_timing.enabled &&
                full_batch && full_batches >= 2) {
                unsigned int exp_arg = mystuff->exponent, bits_arg = (unsigned int)mystuff->gpu_sieve_processing_size;
                int bit_max64 = mystuff->bit_min - 63, shift_arg = shiftcount;
                unsigned int *bit_array = mystuff->d_bitarray[buf], *res_arg = mystuff->d_RES;
                int192 b_arg            = b_preinit;
                int96 k_arg             = k_base;
                void *tf_args[]         = { &exp_arg, &k_arg, &bit_array, &bits_arg, &shift_arg, &b_arg, &res_arg, &bit_max64 };
                const gs_variant_t *gv  = &MFAKTC_GS_VARIANTS[variant];
                size_t smem             = gs_variant_shared_mem(gv, shared_mem_required);
                cudaError_t graph_err = cudaSuccess;

                graph_err = configure_smem_for(variant);
                if (graph_err != cudaSuccess) gs_stream_capture_recover(sieve_stream, tf_stream);

                if (graph_err == cudaSuccess && gs_batch_graph &&
                    (gs_graph_nblocks != numblocks || gs_graph_smem != (int)smem || gs_graph_kernel != gv->kernel ||
                                       gs_graph_pipeline != pipeline)) {
                    cudaGraphExecDestroy(gs_batch_graph);
                    gs_batch_graph = NULL;
                }
                if (graph_err == cudaSuccess && gs_batch_graph == NULL) {
                    cudaGraph_t graph = NULL;
                    int capturing     = 0;

                    if (pipeline) {
                        graph_err = cudaStreamSynchronize(sieve_stream);
                        if (graph_err == cudaSuccess) graph_err = cudaStreamSynchronize(tf_stream);
                    }
                    if (graph_err == cudaSuccess)
                        graph_err = cudaStreamBeginCapture(sieve_stream, cudaStreamCaptureModeGlobal);
                    if (graph_err == cudaSuccess) capturing = 1;
                    if (graph_err == cudaSuccess && pipeline)
                        graph_err = cudaStreamWaitEvent(sieve_stream, mystuff->gs_tf_done[buf], 0);
                    if (graph_err == cudaSuccess) {
                        gpusieve_launch(mystuff, k_remaining, sieve_stream, bit_array);
                        graph_err = cudaGetLastError();
                    }
                    if (graph_err == cudaSuccess && pipeline)
                        graph_err = cudaEventRecord(mystuff->gs_sieve_done[buf], sieve_stream);
                    if (graph_err == cudaSuccess && pipeline)
                        graph_err = cudaStreamWaitEvent(tf_stream, mystuff->gs_sieve_done[buf], 0);
                    if (graph_err == cudaSuccess)
                        graph_err = gs_launch_variant(gv, numblocks, smem, tf_stream, exp_arg, k_arg, bit_array, bits_arg,
                                                      shift_arg, b_arg, res_arg, bit_max64);
                    if (graph_err == cudaSuccess && pipeline)
                        graph_err = cudaEventRecord(mystuff->gs_tf_done[buf], tf_stream);
                    if (capturing) {
                        cudaError_t end_err = cudaStreamEndCapture(sieve_stream, &graph);
                        if (graph_err == cudaSuccess) graph_err = end_err;
                    }
                    if (graph_err == cudaSuccess) graph_err = gs_graph_find_kernels(graph, &gs_graph_sieve_node, &gs_graph_tf_node);
                    if (graph_err == cudaSuccess) graph_err = cudaGraphInstantiate(&gs_batch_graph, graph, NULL, NULL, 0);
                    if (graph) cudaGraphDestroy(graph);
                    if (graph_err == cudaSuccess) {
                        gs_graph_nblocks  = numblocks;
                        gs_graph_smem     = (int)smem;
                        gs_graph_kernel   = gv->kernel;
                        gs_graph_pipeline = pipeline;
                    } else if (gs_batch_graph) {
                        cudaGraphExecDestroy(gs_batch_graph);
                        gs_batch_graph = NULL;
                    }
                } else if (graph_err == cudaSuccess && gs_batch_graph != NULL) {
                    cudaKernelNodeParams sp, tp;
                    gpusieve_graph_sieve_params(mystuff, (int)k_remaining, bit_array, &sp);
                    gs_fill_tf_node(&tp, gv, numblocks, smem, tf_args);
                    graph_err = cudaGraphExecKernelNodeSetParams(gs_batch_graph, gs_graph_sieve_node, &sp);
                    if (graph_err == cudaSuccess)
                        graph_err = cudaGraphExecKernelNodeSetParams(gs_batch_graph, gs_graph_tf_node, &tp);
                }
                if (graph_err == cudaSuccess && gs_batch_graph)
                    graph_err = cudaGraphLaunch(gs_batch_graph, sieve_stream);
                if (graph_err == cudaSuccess && gs_batch_graph)
                    launched_graph = 1;
                else if (graph_err != cudaSuccess) {
                    gs_stream_capture_recover(sieve_stream, tf_stream);
                    if (gs_batch_graph) {
                        cudaGraphExecDestroy(gs_batch_graph);
                        gs_batch_graph = NULL;
                    }
                    gs_graph_disabled = 1;
                    if (mystuff->verbosity >= 1)
                        printf("WARNING: CUDA graph replay disabled (%s); using ordinary launches\n", cudaGetErrorString(graph_err));
                }
            }
        }
#endif
        if (launched_graph) {
            if (pipeline) have_tf[buf] = 1;
        } else {
        if (pipeline && have_tf[buf]) {
            cuda_ret = cudaStreamWaitEvent(sieve_stream, mystuff->gs_tf_done[buf], 0);
            if (cuda_ret != cudaSuccess) {
                printf("ERROR: GPU-sieve pipeline wait failed: %s\n", cudaGetErrorString(cuda_ret));
                return RET_CUDA_ERROR;
            }
        }

        cuda_ret = gs_timing_begin_batch(&gs_class_timing);
        if (cuda_ret == cudaSuccess) cuda_ret = gs_timing_record(&gs_class_timing, GS_EV_SIEVE_START, sieve_stream);
        if (cuda_ret != cudaSuccess) {
            printf("ERROR: GPU timing failed: %s\n", cudaGetErrorString(cuda_ret));
            return RET_CUDA_ERROR;
        }
        gpusieve_launch(mystuff, k_remaining, sieve_stream, mystuff->d_bitarray[buf]);
        cuda_ret = cudaGetLastError();
        if (cuda_ret != cudaSuccess) {
            printf("ERROR: GPU sieve launch failed: %s\n", cudaGetErrorString(cuda_ret));
            return RET_CUDA_ERROR;
        }
        gs_timing_record(&gs_class_timing, GS_EV_SIEVE_END, sieve_stream);
        if (pipeline) {
            cuda_ret = cudaEventRecord(mystuff->gs_sieve_done[buf], sieve_stream);
            if (cuda_ret != cudaSuccess) {
                printf("ERROR: GPU-sieve pipeline event failed: %s\n", cudaGetErrorString(cuda_ret));
                return RET_CUDA_ERROR;
            }
        }

#ifdef MFAKTC_GS_VARIANTS
        if (mystuff->mode == MODE_NORMAL && !mystuff->gpu_sieve_autotune) {
            if (pipeline) cudaEventSynchronize(mystuff->gs_sieve_done[buf]);
            if (tuned_device >= 0 && tuned_bit_min == mystuff->bit_min && tuned_exponent == mystuff->exponent)
                variant = tuned_variant;
            else if (numblocks == mystuff->gpu_sieve_size / mystuff->gpu_sieve_processing_size) {
                auto launch = [&](int a, unsigned int *result) {
                    const gs_variant_t *gv = &MFAKTC_GS_VARIANTS[applicable[a]];
                    return gs_launch_variant(gv, numblocks, gs_variant_shared_mem(gv, shared_mem_required), 0, mystuff->exponent,
                                             k_base, mystuff->d_bitarray[0], mystuff->gpu_sieve_processing_size, shiftcount, b_preinit,
                                             result, mystuff->bit_min - 63);
                };
                int fastest = 0;
                cuda_ret    = mfaktc_time_variants(num_applicable, launch, &fastest);
                if (cuda_ret != cudaSuccess) {
                    printf("ERROR: TF kernel calibration failed: %s\n", cudaGetErrorString(cuda_ret));
                    return RET_CUDA_ERROR;
                }
                tuned_variant  = applicable[fastest];
                cudaGetDevice(&tuned_device);
                tuned_bit_min  = mystuff->bit_min;
                tuned_cat      = gs_exp_category(mystuff->exponent);
                tuned_exponent = mystuff->exponent;
                variant        = tuned_variant;
                if (mystuff->verbosity >= 2)
                    printf("Timed TF launch selection: %d threads/block, %d candidate%s/thread, %s candidate list, %s kernel\n",
                           MFAKTC_GS_VARIANTS[variant].threads, MFAKTC_GS_VARIANTS[variant].cands,
                           MFAKTC_GS_VARIANTS[variant].cands == 1 ? "" : "s",
                           MFAKTC_GS_VARIANTS[variant].warp_local ? "warp-local" : "block",
                           MFAKTC_GS_VARIANTS[variant].bit_max64 ? "bit-level" : "generic");
            }
        }
#endif

        // Now let the GPU trial factor the candidates that survived the sieving

        if (pipeline) {
            cuda_ret = cudaStreamWaitEvent(tf_stream, mystuff->gs_sieve_done[buf], 0);
            if (cuda_ret != cudaSuccess) {
                printf("ERROR: GPU-sieve pipeline wait failed: %s\n", cudaGetErrorString(cuda_ret));
                return RET_CUDA_ERROR;
            }
        }
        gs_timing_record(&gs_class_timing, GS_EV_TF_START, tf_stream);
#ifdef MFAKTC_GS_VARIANTS
        cuda_ret = configure_smem_for(variant);
        if (cuda_ret != cudaSuccess) {
            printf("ERROR: could not configure TF shared memory: %s\n", cudaGetErrorString(cuda_ret));
            return RET_CUDA_ERROR;
        }
        cuda_ret = gs_launch_variant(&MFAKTC_GS_VARIANTS[variant], numblocks,
                                     gs_variant_shared_mem(&MFAKTC_GS_VARIANTS[variant], shared_mem_required), tf_stream,
                                     mystuff->exponent, k_base, mystuff->d_bitarray[buf], mystuff->gpu_sieve_processing_size,
                                     shiftcount, b_preinit, mystuff->d_RES, mystuff->bit_min - 63);
        if (cuda_ret != cudaSuccess) {
            printf("ERROR: TF kernel launch failed: %s\n", cudaGetErrorString(cuda_ret));
            return RET_CUDA_ERROR;
        }
#else
        MFAKTC_FUNC<<<numblocks, threads_per_block, shared_mem_required, tf_stream>>>(
            mystuff->exponent, k_base, mystuff->d_bitarray[buf], mystuff->gpu_sieve_processing_size, shiftcount, b_preinit, mystuff->d_RES
#if defined(TF_BARRETT) && \
    (defined(TF_BARRETT_87BIT_GS) || defined(TF_BARRETT_88BIT_GS) || defined(TF_BARRETT_92BIT_GS) || defined(DEBUG_GPU_MATH))
            ,
            mystuff->bit_min - 63
#endif
#ifdef DEBUG_GPU_MATH
            ,
            mystuff->d_modbasecase_debug
#endif
        );
#endif
        gs_timing_record(&gs_class_timing, GS_EV_TF_END, tf_stream);
        gs_timing_end_batch(&gs_class_timing);
        if (pipeline) {
            cuda_ret = cudaEventRecord(mystuff->gs_tf_done[buf], tf_stream);
            if (cuda_ret != cudaSuccess) {
                printf("ERROR: GPU-sieve pipeline event failed: %s\n", cudaGetErrorString(cuda_ret));
                return RET_CUDA_ERROR;
            }
            have_tf[buf] = 1;
        }
        } // !launched_graph

        // Count the number of blocks processed
        count += numblocks;

        // Move to next batch of k's
        k_min += (unsigned long long)mystuff->gpu_sieve_batch_size * NUM_CLASSES;
        if (k_min > k_max) break;

        gpusieve_advance_class_on(mystuff, mystuff->gpu_sieve_batch_size, sieve_stream);
        if (pipeline) buf = 1 - buf;
    }

    /* download results from GPU */
    if (mystuff->gpu_sieve_pipeline && mystuff->gs_tf_stream) {
        cudaStreamSynchronize(mystuff->gs_sieve_stream);
        cudaStreamSynchronize(mystuff->gs_tf_stream);
    }
    cuda_ret = cudaMemcpy(mystuff->h_RES, mystuff->d_RES, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess) {
        printf("ERROR: GPU-sieve trial factoring failed: %s\n", cudaGetErrorString(cuda_ret));
        return RET_CUDA_ERROR;
    }

    // An overflowing block skipped all its candidates, so the whole class must be
    // redone from its original k_min.  The class timer keeps running across the retry.
    if (mystuff->h_RES[31] != 0 && !conservative_candidate_buffer) {
        if (mystuff->verbosity >= 1)
            printf("GPU sieve block had %u candidates, more than its buffer of %d holds; retrying class with a larger buffer\n",
                   mystuff->h_RES[31], shared_mem_required / (int)sizeof(unsigned short));
        conservative_candidate_buffer = 1;
#ifdef MFAKTC_GS_VARIANTS
        if (gs_batch_graph) {
            cudaGraphExecDestroy(gs_batch_graph);
            gs_batch_graph = NULL;
        }
#endif
        goto restart_class;
    }

#ifdef DEBUG_GPU_MATH
    cudaMemcpy(mystuff->h_modbasecase_debug, mystuff->d_modbasecase_debug, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    for (i = 0; i < 32; i++)
        if (mystuff->h_modbasecase_debug[i] != 0) printf("h_modbasecase_debug[%2d] = %u\n", i, mystuff->h_modbasecase_debug[i]);
#endif

    // Set grid count to the number of blocks processed.  The print code will convert this to a
    // count of candidates processed (by multiplying by 8192 * THREADS_PER_BLOCK.
    // This count isn't an exact match to CPU sieving case as that counts candidates after sieving
    // and we are counting candidates before sieving.  We'd have to modify the TF kernels to count
    // the candidates processed to be completely compatible.
    mystuff->stats.grid_count = count;

    // Keep track of time spent TFing this class and this bitlevel
    /* prevent division by zero if timer resolution is too low */
    mystuff->stats.class_time = timer_diff(&timer) / 1000;
    mystuff->stats.bit_level_time += timer_diff(&timer) / 1000;
    if (mystuff->stats.class_time == 0) mystuff->stats.class_time = 1;

    // GPU sieving does not wait on the CPU (also used by print_status_line to indicate this is a GPU sieving kernel)
    mystuff->stats.cpu_wait = -2.0f;

    cuda_ret = gs_timing_end_class(&gs_class_timing);
    if (cuda_ret != cudaSuccess) {
        printf("ERROR: GPU timing failed: %s\n", cudaGetErrorString(cuda_ret));
        return RET_CUDA_ERROR;
    }
    if (gs_class_timing.enabled)
        printf("GPU time for class %d: sieve %.2f ms, TF %.2f ms (%d batches); class %llu ms\n", mystuff->stats.class_number,
               gs_class_timing.sieve_ms, gs_class_timing.tf_ms, gs_class_timing.batches, mystuff->stats.class_time);

    // Print out a useful status line
    print_status_line(mystuff);

    if (mystuff->h_RES[31] != 0) {
        logprintf(
            mystuff,
            "ERROR: a GPU sieve block had %u candidates, more than its buffer of %d holds. Factors could have been missed due to untested candidates. Please report this.\n",
            mystuff->h_RES[31], shared_mem_required / (int)sizeof(unsigned short));
        return RET_CUDA_ERROR;
    }

    // Print out any found factors
    factorsfound = mystuff->h_RES[0];
    for (i = 0; (i < factorsfound) && (i < 10); i++) {
        factor.d2 = mystuff->h_RES[i * 3 + 1];
        factor.d1 = mystuff->h_RES[i * 3 + 2];
        factor.d0 = mystuff->h_RES[i * 3 + 3];
        print_dez96(factor, string);
        print_factor(mystuff, i, string);
    }
    if (factorsfound >= 10) {
        print_factor(mystuff, factorsfound, NULL);
    }

    return factorsfound;
}

#undef MFAKTC_FUNC
#ifdef MFAKTC_GS_VARIANTS
#undef MFAKTC_GS_VARIANTS
#undef MFAKTC_GS_FAMILY
#endif
