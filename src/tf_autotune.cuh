/*
This file is part of mfaktc.

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

#ifndef MFAKTC_TF_AUTOTUNE_CUH
#define MFAKTC_TF_AUTOTUNE_CUH

#include <cuda_runtime.h>

#define GS_MAX_VARIANTS 64

/* Every timed variant of a GPU-sieve Barrett kernel has this signature. */
typedef void (*gs_kernel_t)(unsigned int exp, int96 k_base, const unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                            int192 b_preinit, unsigned int *RES, int bit_max64);

typedef struct {
    gs_kernel_t kernel;
    int threads;    /* block size the kernel was compiled for */
    int cands;      /* factor candidates each thread tests at once */
    int warp_local; /* 1: warp-local candidate extraction, no dynamic shared memory */
    int bit_max64;  /* 0, or the only bit_max64 the kernel was compiled for */
} gs_variant_t;

static size_t gs_variant_shared_mem(const gs_variant_t *variant, int shared_mem_required)
{
    return variant->warp_local ? 0 : (size_t)shared_mem_required;
}

/* Lists the variants that can run with this bit_max64 (all generic ones and
   those compiled for it) in applicable[] and returns their number. */
static int gs_applicable_variants(const gs_variant_t *variants, int count, int bit_max64, int *applicable)
{
    int n = 0;

    for (int v = 0; v < count && n < GS_MAX_VARIANTS; v++)
        if (variants[v].bit_max64 == 0 || variants[v].bit_max64 == bit_max64) applicable[n++] = v;
    return n;
}

static cudaError_t gs_launch_variant(const gs_variant_t *variant, int numblocks, size_t shared_mem, cudaStream_t stream, unsigned int exp,
                                     int96 k_base, const unsigned int *bit_array, unsigned int bits_to_process, int shiftcount,
                                     int192 b_preinit, unsigned int *RES, int bit_max64)
{
    void *args[] = { &exp, &k_base, &bit_array, &bits_to_process, &shiftcount, &b_preinit, &RES, &bit_max64 };
    return cudaLaunchKernel((const void *)variant->kernel, dim3(numblocks), dim3(variant->threads), args, shared_mem, stream);
}

/* Times count launch variants of the same TF work on the current sieve batch
   and sets *best to the index of the fastest.  launch(index, result) must
   start variant index writing to the result array it is given; that is a
   scratch array, so calibration never adds factors to the real results.  All
   variants are warmed up first and the measurement order is reversed on
   alternate passes to reduce ordering bias. */
template <typename Launch> static cudaError_t mfaktc_time_variants(int count, Launch launch, int *best)
{
    const int repeats     = 4;
    unsigned int *scratch = NULL;
    cudaEvent_t start = NULL, stop = NULL;
    float elapsed, sums[GS_MAX_VARIANTS];
    cudaError_t err, err2;

    if (count < 1 || count > GS_MAX_VARIANTS) return cudaErrorInvalidValue;
    for (int v = 0; v < count; v++)
        sums[v] = 0.0f;

    err = cudaMalloc((void **)&scratch, 32 * sizeof(unsigned int));
    if (err == cudaSuccess) err = cudaEventCreate(&start);
    if (err == cudaSuccess) err = cudaEventCreate(&stop);

    for (int pass = -1; pass < repeats && err == cudaSuccess; pass++) {
        for (int position = 0; position < count && err == cudaSuccess; position++) {
            const int variant = (pass & 1) ? count - 1 - position : position;

            err = cudaMemsetAsync(scratch, 0, 32 * sizeof(unsigned int), 0);
            if (err == cudaSuccess) err = cudaEventRecord(start, 0);
            if (err == cudaSuccess) err = launch(variant, scratch);
            if (err == cudaSuccess) err = cudaEventRecord(stop, 0);
            if (err == cudaSuccess) err = cudaEventSynchronize(stop);
            if (err == cudaSuccess) err = cudaEventElapsedTime(&elapsed, start, stop);
            if (err == cudaSuccess && pass >= 0) sums[variant] += elapsed; // pass -1 is the warm-up
        }
    }
    if (err == cudaSuccess) {
        *best = 0;
        for (int v = 1; v < count; v++)
            if (sums[v] < sums[*best]) *best = v;
    }

    if (stop != NULL && (err2 = cudaEventDestroy(stop)) != cudaSuccess && err == cudaSuccess) err = err2;
    if (start != NULL && (err2 = cudaEventDestroy(start)) != cudaSuccess && err == cudaSuccess) err = err2;
    if (scratch != NULL && (err2 = cudaFree(scratch)) != cudaSuccess && err == cudaSuccess) err = err2;
    return err;
}

#endif
