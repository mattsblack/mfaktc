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
#endif
#elif defined TF_BARRETT_77BIT_GS
    extern "C" __host__ int tf_class_barrett77_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_barrett77_gs
#ifndef DEBUG_GPU_MATH
#define MFAKTC_GS_VARIANTS gs_variants_barrett77
#endif
#elif defined TF_BARRETT_79BIT_GS
extern "C" __host__ int tf_class_barrett79_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_barrett79_gs
#ifndef DEBUG_GPU_MATH
#define MFAKTC_GS_VARIANTS gs_variants_barrett79
#endif
#elif defined TF_BARRETT_87BIT_GS
extern "C" __host__ int tf_class_barrett87_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_barrett87_gs
#ifndef DEBUG_GPU_MATH
#define MFAKTC_GS_VARIANTS gs_variants_barrett87
#endif
#elif defined TF_BARRETT_88BIT_GS
extern "C" __host__ int tf_class_barrett88_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_barrett88_gs
#ifndef DEBUG_GPU_MATH
#define MFAKTC_GS_VARIANTS gs_variants_barrett88
#endif
#else
extern "C" __host__ int tf_class_barrett92_gs(unsigned long long int k_min, unsigned long long int k_max, mystuff_t *mystuff)
#define MFAKTC_FUNC mfaktc_barrett92_gs
#ifndef DEBUG_GPU_MATH
#define MFAKTC_GS_VARIANTS gs_variants_barrett92
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
    k_min = original_k_min;
    count = 0;

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
            // occupancy calculator about a launch that uses it.
            cuda_ret = cudaFuncSetAttribute(gv->kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_required);
            if (cuda_ret != cudaSuccess) {
                printf("ERROR: could not configure %d bytes of dynamic shared memory: %s\n", shared_mem_required,
                       cudaGetErrorString(cuda_ret));
                return RET_CUDA_ERROR;
            }
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

    // Loop until all the k's are processed
    for (;;) {
        // Calculate the number of k's remaining.  Round this up so that we sieve an array that is
        // a multiple of the bits processed by each TF kernel (my_stuff->gpu_sieve_processing_size).

        k_remaining = ((k_max - k_min + 1) + NUM_CLASSES - 1) / NUM_CLASSES;
        if (k_remaining < (unsigned long long)mystuff->gpu_sieve_size) {
            numblocks   = ((int)k_remaining + mystuff->gpu_sieve_processing_size - 1) / mystuff->gpu_sieve_processing_size;
            k_remaining = numblocks * mystuff->gpu_sieve_processing_size;
        } else
            numblocks = mystuff->gpu_sieve_size / mystuff->gpu_sieve_processing_size;

        // Do some sieving on the GPU.

        cuda_ret = gs_timing_begin_batch(&gs_class_timing);
        if (cuda_ret == cudaSuccess) cuda_ret = gs_timing_record(&gs_class_timing, GS_EV_SIEVE_START, 0);
        if (cuda_ret != cudaSuccess) {
            printf("ERROR: GPU timing failed: %s\n", cudaGetErrorString(cuda_ret));
            return RET_CUDA_ERROR;
        }
        gpusieve(mystuff, k_remaining);
        gs_timing_record(&gs_class_timing, GS_EV_SIEVE_END, 0);

        // Set the k value corresponding to the first bit in the bit array

        k_base.d0 = (int)(k_min & 0xFFFFFFFF);
        k_base.d1 = (int)(k_min >> 32);
        k_base.d2 = 0;

#ifdef MFAKTC_GS_VARIANTS
        // Higher occupancy is not necessarily faster, so time all independently
        // compiled variants on the first full sieve batch and keep the fastest.
        // The arithmetic workload depends on the exponent and bit level, so
        // recalibrate when those or the launch resources change.
        static int tuned_device = -1, tuned_smem = -1, tuned_process_size = -1, tuned_bit_min = -1;
        static unsigned int tuned_exponent = 0;
        static int tuned_variant = 0;
        int current_device = -1;

        cuda_ret = cudaGetDevice(&current_device);
        if (cuda_ret != cudaSuccess) {
            printf("ERROR: cudaGetDevice() failed: %s\n", cudaGetErrorString(cuda_ret));
            return RET_CUDA_ERROR;
        }
        if (mystuff->mode != MODE_NORMAL) {
            // keep the self-test's variant
        } else if (tuned_device == current_device && tuned_smem == shared_mem_required &&
                   tuned_process_size == mystuff->gpu_sieve_processing_size && tuned_exponent == mystuff->exponent &&
                   tuned_bit_min == mystuff->bit_min) {
            variant = tuned_variant;
        } else if (numblocks == mystuff->gpu_sieve_size / mystuff->gpu_sieve_processing_size) {
            // Only a full batch is a representative sample; short batches keep the occupancy-based choice.
            auto launch = [&](int a, unsigned int *result) {
                const gs_variant_t *gv = &MFAKTC_GS_VARIANTS[applicable[a]];
                return gs_launch_variant(gv, numblocks, gs_variant_shared_mem(gv, shared_mem_required), 0, mystuff->exponent, k_base,
                                         mystuff->d_bitarray, mystuff->gpu_sieve_processing_size, shiftcount, b_preinit, result,
                                         mystuff->bit_min - 63);
            };
            int fastest = 0;
            cuda_ret    = mfaktc_time_variants(num_applicable, launch, &fastest);
            if (cuda_ret != cudaSuccess) {
                printf("ERROR: TF kernel calibration failed: %s\n", cudaGetErrorString(cuda_ret));
                return RET_CUDA_ERROR;
            }
            tuned_variant      = applicable[fastest];
            tuned_device       = current_device;
            tuned_smem         = shared_mem_required;
            tuned_process_size = mystuff->gpu_sieve_processing_size;
            tuned_exponent     = mystuff->exponent;
            tuned_bit_min      = mystuff->bit_min;
            variant            = tuned_variant;
            if (mystuff->verbosity >= 2)
                printf("Timed TF launch selection: %d threads/block, %d candidate%s/thread, %s candidate list, %s kernel\n",
                       MFAKTC_GS_VARIANTS[variant].threads, MFAKTC_GS_VARIANTS[variant].cands,
                       MFAKTC_GS_VARIANTS[variant].cands == 1 ? "" : "s", MFAKTC_GS_VARIANTS[variant].warp_local ? "warp-local" : "block",
                       MFAKTC_GS_VARIANTS[variant].bit_max64 ? "bit-level" : "generic");
        }
#endif

        // Now let the GPU trial factor the candidates that survived the sieving

        gs_timing_record(&gs_class_timing, GS_EV_TF_START, 0);
#ifdef MFAKTC_GS_VARIANTS
        cuda_ret = gs_launch_variant(&MFAKTC_GS_VARIANTS[variant], numblocks,
                                     gs_variant_shared_mem(&MFAKTC_GS_VARIANTS[variant], shared_mem_required), 0, mystuff->exponent,
                                     k_base, mystuff->d_bitarray, mystuff->gpu_sieve_processing_size, shiftcount, b_preinit,
                                     mystuff->d_RES, mystuff->bit_min - 63);
        if (cuda_ret != cudaSuccess) {
            printf("ERROR: TF kernel launch failed: %s\n", cudaGetErrorString(cuda_ret));
            return RET_CUDA_ERROR;
        }
#else
        MFAKTC_FUNC<<<numblocks, threads_per_block, shared_mem_required>>>(
            mystuff->exponent, k_base, mystuff->d_bitarray, mystuff->gpu_sieve_processing_size, shiftcount, b_preinit, mystuff->d_RES
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
        gs_timing_record(&gs_class_timing, GS_EV_TF_END, 0);
        gs_timing_end_batch(&gs_class_timing);

        // Count the number of blocks processed
        count += numblocks;

        // Move to next batch of k's
        k_min += (unsigned long long)mystuff->gpu_sieve_size * NUM_CLASSES;
        if (k_min > k_max) break;

        gpusieve_advance_class(mystuff, mystuff->gpu_sieve_size);
    }

    /* download results from GPU */
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
#endif
