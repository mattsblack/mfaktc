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

/* Times two launch configurations of the same TF kernel on the current sieve
   batch and sets *best_threads to 128 or 256, whichever ran faster.  The launch
   functors take the result array to write to; they are given a scratch array so
   calibration never adds factors to the real results.  Both variants are warmed
   up first and the measurement order alternates to reduce ordering bias. */
template <typename Launch128, typename Launch256>
static cudaError_t mfaktc_time_block_variants(Launch128 launch128, Launch256 launch256, int *best_threads)
{
    const int repeats      = 4;
    unsigned int *scratch  = NULL;
    cudaEvent_t start      = NULL, stop = NULL;
    float elapsed, sums[2] = { 0.0f, 0.0f };
    cudaError_t err, err2;

    err = cudaMalloc((void **)&scratch, 32 * sizeof(unsigned int));
    if (err == cudaSuccess) err = cudaEventCreate(&start);
    if (err == cudaSuccess) err = cudaEventCreate(&stop);

    for (int pass = -1; pass < repeats && err == cudaSuccess; pass++) {
        for (int position = 0; position < 2 && err == cudaSuccess; position++) {
            const int variant = pass < 0 ? position : position ^ (pass & 1);

            err = cudaMemsetAsync(scratch, 0, 32 * sizeof(unsigned int), 0);
            if (err == cudaSuccess) err = cudaEventRecord(start, 0);
            if (err != cudaSuccess) break;
            if (variant == 0)
                launch128(scratch);
            else
                launch256(scratch);
            err = cudaGetLastError();
            if (err == cudaSuccess) err = cudaEventRecord(stop, 0);
            if (err == cudaSuccess) err = cudaEventSynchronize(stop);
            if (err == cudaSuccess) err = cudaEventElapsedTime(&elapsed, start, stop);
            if (err == cudaSuccess && pass >= 0) sums[variant] += elapsed; // pass -1 is the warm-up
        }
    }
    if (err == cudaSuccess) *best_threads = sums[0] < sums[1] ? 128 : 256;

    if (stop != NULL && (err2 = cudaEventDestroy(stop)) != cudaSuccess && err == cudaSuccess) err = err2;
    if (start != NULL && (err2 = cudaEventDestroy(start)) != cudaSuccess && err == cudaSuccess) err = err2;
    if (scratch != NULL && (err2 = cudaFree(scratch)) != cudaSuccess && err == cudaSuccess) err = err2;
    return err;
}

#endif
