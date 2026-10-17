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

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "params.h"
#include "my_types.h"
#include "compatibility.h"
#include "my_intrinsics.h"

#define NVCC_EXTERN
#include "sieve.h"
#include "timer.h"
#include "output.h"
#undef NVCC_EXTERN

#include "tf_debug.h"
#include "tf_96bit_base_math.cu"
#include "tf_96bit_helper.cu"

#undef INV_160_96
#include "tf_barrett96_div.cu"
#define INV_160_96
#include "tf_barrett96_div.cu"
#undef INV_160_96

#include "tf_barrett96_core.cu"

#include "gpusieve_helper.cu"
#include "tf_autotune.cuh"

#define KERNEL_MIN_BLOCKS 2

__global__ void
#ifndef DEBUG_GPU_MATH
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett92_gs(unsigned int exp, int96 k_base, const unsigned int *__restrict__ bit_array,
                        unsigned int bits_to_process, int shiftcount, int192 b_preinit, unsigned int *__restrict__ RES, int bit_max64)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett92_gs(unsigned int exp, int96 k_base, const unsigned int *__restrict__ bit_array,
                        unsigned int bits_to_process, int shiftcount, int192 b_preinit, unsigned int *__restrict__ RES, int bit_max64,
                        unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
bit_max64 is the number of bits in the factor (minus 64)
*/
{
    int96 f, f_base;
    int i, initial_shifter_value, total_bit_count, k_delta;
    extern __shared__ unsigned short k_deltas[]; // Write bits to test here.  Launching program must estimate
    // how much shared memory to allocate based on number of primes sieved.

    create_k_deltas(bit_array, bits_to_process, &total_bit_count, k_deltas, RES);
    create_fbase96(&f_base, k_base, exp, bits_to_process);

    initial_shifter_value = exp << (32 - shiftcount); // Initial shifter value

    // Loop til the k values written to shared memory are exhausted
    for (i = threadIdx.x; i < total_bit_count; i += blockDim.x) {
        // Get the (k - k_base) value to test
        k_delta = k_deltas[i];

        // Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.
        f.d0 = __add_cc(f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
        f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
        f.d2 = __addc(f_base.d2, 0);

        test_FC96_barrett92(f, b_preinit, initial_shifter_value, RES, bit_max64
#ifdef DEBUG_GPU_MATH
                            ,
                            modbasecase_debug
#endif
        );
    }
}

__global__ void
#ifndef DEBUG_GPU_MATH
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett88_gs(unsigned int exp, int96 k_base, const unsigned int *__restrict__ bit_array,
                        unsigned int bits_to_process, int shiftcount, int192 b_preinit, unsigned int *__restrict__ RES, int bit_max64)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett88_gs(unsigned int exp, int96 k_base, const unsigned int *__restrict__ bit_array,
                        unsigned int bits_to_process, int shiftcount, int192 b_preinit, unsigned int *__restrict__ RES, int bit_max64,
                        unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
bit_max64 is the number of bits in the factor (minus 64)
*/
{
    int96 f, f_base;
    int i, initial_shifter_value, total_bit_count, k_delta;
    extern __shared__ unsigned short k_deltas[]; // Write bits to test here.  Launching program must estimate
    // how much shared memory to allocate based on number of primes sieved.

    create_k_deltas(bit_array, bits_to_process, &total_bit_count, k_deltas, RES);
    create_fbase96(&f_base, k_base, exp, bits_to_process);

    initial_shifter_value = exp << (32 - shiftcount); // Initial shifter value

    // Loop til the k values written to shared memory are exhausted
    for (i = threadIdx.x; i < total_bit_count; i += blockDim.x) {
        // Get the (k - k_base) value to test
        k_delta = k_deltas[i];

        // Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.
        f.d0 = __add_cc(f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
        f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
        f.d2 = __addc(f_base.d2, 0);

        test_FC96_barrett88(f, b_preinit, initial_shifter_value, RES, bit_max64
#ifdef DEBUG_GPU_MATH
                            ,
                            modbasecase_debug
#endif
        );
    }
}

__global__ void
#ifndef DEBUG_GPU_MATH
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett87_gs(unsigned int exp, int96 k_base, const unsigned int *__restrict__ bit_array,
                        unsigned int bits_to_process, int shiftcount, int192 b_preinit, unsigned int *__restrict__ RES, int bit_max64)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett87_gs(unsigned int exp, int96 k_base, const unsigned int *__restrict__ bit_array,
                        unsigned int bits_to_process, int shiftcount, int192 b_preinit, unsigned int *__restrict__ RES, int bit_max64,
                        unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
bit_max64 is the number of bits in the factor (minus 64)
*/
{
    int96 f, f_base;
    int i, initial_shifter_value, total_bit_count, k_delta;
    extern __shared__ unsigned short k_deltas[]; // Write bits to test here.  Launching program must estimate
    // how much shared memory to allocate based on number of primes sieved.

    create_k_deltas(bit_array, bits_to_process, &total_bit_count, k_deltas, RES);
    create_fbase96(&f_base, k_base, exp, bits_to_process);

    initial_shifter_value = exp << (32 - shiftcount); // Initial shifter value

    // Loop til the k values written to shared memory are exhausted
    for (i = threadIdx.x; i < total_bit_count; i += blockDim.x) {
        // Get the (k - k_base) value to test
        k_delta = k_deltas[i];

        // Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.
        f.d0 = __add_cc(f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
        f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
        f.d2 = __addc(f_base.d2, 0);

        test_FC96_barrett87(f, b_preinit, initial_shifter_value, RES, bit_max64
#ifdef DEBUG_GPU_MATH
                            ,
                            modbasecase_debug
#endif
        );
    }
}

__global__ void
#ifndef DEBUG_GPU_MATH
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett79_gs(unsigned int exp, int96 k_base, const unsigned int *__restrict__ bit_array,
                        unsigned int bits_to_process, int shiftcount, int192 b_preinit, unsigned int *__restrict__ RES)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett79_gs(unsigned int exp, int96 k_base, const unsigned int *__restrict__ bit_array,
                        unsigned int bits_to_process, int shiftcount, int192 b_preinit, unsigned int *__restrict__ RES, int bit_max64,
                        unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
    int96 f, f_base;
    int i, initial_shifter_value, total_bit_count, k_delta;
    extern __shared__ unsigned short k_deltas[]; // Write bits to test here.  Launching program must estimate
    // how much shared memory to allocate based on number of primes sieved.

    create_k_deltas(bit_array, bits_to_process, &total_bit_count, k_deltas, RES);
    create_fbase96(&f_base, k_base, exp, bits_to_process);

    initial_shifter_value = exp << (32 - shiftcount); // Initial shifter value

    // Loop til the k values written to shared memory are exhausted
    for (i = threadIdx.x; i < total_bit_count; i += blockDim.x) {
        // Get the (k - k_base) value to test
        k_delta = k_deltas[i];

        // Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.
        f.d0 = __add_cc(f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
        f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
        f.d2 = __addc(f_base.d2, 0);

        test_FC96_barrett79(f, b_preinit, initial_shifter_value, RES
#ifdef DEBUG_GPU_MATH
                            ,
                            bit_max64, modbasecase_debug
#endif
        );
    }
}

__global__ void
#ifndef DEBUG_GPU_MATH
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett77_gs(unsigned int exp, int96 k_base, const unsigned int *__restrict__ bit_array,
                        unsigned int bits_to_process, int shiftcount, int192 b_preinit, unsigned int *__restrict__ RES)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett77_gs(unsigned int exp, int96 k_base, const unsigned int *__restrict__ bit_array,
                        unsigned int bits_to_process, int shiftcount, int192 b_preinit, unsigned int *__restrict__ RES, int bit_max64,
                        unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
    int96 f, f_base;
    int i, initial_shifter_value, total_bit_count, k_delta;
    extern __shared__ unsigned short k_deltas[]; // Write bits to test here.  Launching program must estimate
    // how much shared memory to allocate based on number of primes sieved.

    create_k_deltas(bit_array, bits_to_process, &total_bit_count, k_deltas, RES);
    create_fbase96(&f_base, k_base, exp, bits_to_process);

    initial_shifter_value = exp << (32 - shiftcount); // Initial shifter value

    // Loop til the k values written to shared memory are exhausted
    for (i = threadIdx.x; i < total_bit_count; i += blockDim.x) {
        // Get the (k - k_base) value to test
        k_delta = k_deltas[i];

        // Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.
        f.d0 = __add_cc(f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
        f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
        f.d2 = __addc(f_base.d2, 0);

        test_FC96_barrett77(f, b_preinit, initial_shifter_value, RES
#ifdef DEBUG_GPU_MATH
                            ,
                            bit_max64, modbasecase_debug
#endif
        );
    }
}

__global__ void
#ifndef DEBUG_GPU_MATH
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett76_gs(unsigned int exp, int96 k_base, const unsigned int *__restrict__ bit_array,
                        unsigned int bits_to_process, int shiftcount, int192 b_preinit, unsigned int *__restrict__ RES)
#else
__launch_bounds__(THREADS_PER_BLOCK, KERNEL_MIN_BLOCKS)
    mfaktc_barrett76_gs(unsigned int exp, int96 k_base, const unsigned int *__restrict__ bit_array,
                        unsigned int bits_to_process, int shiftcount, int192 b_preinit, unsigned int *__restrict__ RES, int bit_max64,
                        unsigned int *modbasecase_debug)
#endif
/*
computes 2^exp mod f
shiftcount is used for precomputing without mod
a is precomputed on host ONCE.
*/
{
    int96 f, f_base;
    int i, initial_shifter_value, total_bit_count, k_delta;
    extern __shared__ unsigned short k_deltas[]; // Write bits to test here.  Launching program must estimate
    // how much shared memory to allocate based on number of primes sieved.

    create_k_deltas(bit_array, bits_to_process, &total_bit_count, k_deltas, RES);
    create_fbase96(&f_base, k_base, exp, bits_to_process);

    initial_shifter_value = exp << (32 - shiftcount); // Initial shifter value

    // Loop til the k values written to shared memory are exhausted
    for (i = threadIdx.x; i < total_bit_count; i += blockDim.x) {
        // Get the (k - k_base) value to test
        k_delta = k_deltas[i];

        // Compute new f.  This is computed as f = f_base + 2 * (k - k_base) * exp.
        f.d0 = __add_cc(f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
        f.d1 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
        f.d2 = __addc(f_base.d2, 0);

        test_FC96_barrett76(f, b_preinit, initial_shifter_value, RES
#ifdef DEBUG_GPU_MATH
                            ,
                            bit_max64, modbasecase_debug
#endif
        );
    }
}

#ifndef DEBUG_GPU_MATH
#include "tf_barrett96_core_multi.cu"

/* gs_test<BITS, CANDS>::run() tests CANDS candidates with the BITS-bit Barrett
   code.  One candidate per thread uses the original test_FC96_barrettXX(). */
template <int BITS, int CANDS> struct gs_test;

#define GS_BIT_MAX_ARG , bit_max64
#define DEFINE_GS_TEST(BITS, BIT_MAX_ARG)                                                                                          \
    template <int CANDS> struct gs_test<BITS, CANDS> {                                                                             \
        __device__ static __forceinline__ void run(const int96 *f, const int *valid, int192 b, unsigned int shifter, unsigned int *RES, \
                                                   int bit_max64)                                                                  \
        {                                                                                                                          \
            test_FC96_barrett##BITS##_n<CANDS>(f, valid, b, shifter, RES BIT_MAX_ARG);                                             \
        }                                                                                                                          \
    };                                                                                                                             \
    template <> struct gs_test<BITS, 1> {                                                                                          \
        __device__ static __forceinline__ void run(const int96 *f, const int *valid, int192 b, unsigned int shifter, unsigned int *RES, \
                                                   int bit_max64)                                                                  \
        {                                                                                                                          \
            if (valid[0]) test_FC96_barrett##BITS(f[0], b, shifter, RES BIT_MAX_ARG);                                              \
        }                                                                                                                          \
    };

DEFINE_GS_TEST(92, GS_BIT_MAX_ARG)
DEFINE_GS_TEST(88, GS_BIT_MAX_ARG)
DEFINE_GS_TEST(87, GS_BIT_MAX_ARG)
DEFINE_GS_TEST(79, )
DEFINE_GS_TEST(77, )
DEFINE_GS_TEST(76, )
#undef DEFINE_GS_TEST
#undef GS_BIT_MAX_ARG

/* Separate launch-bound variants let NVCC choose an independent register
   budget and schedule for each block size, and with two candidates per thread
   each thread has two independent modular exponentiations to interleave, at
   the cost of more registers.  The host times the variants and keeps the
   fastest.  The 76/77/79-bit variants accept bit_max64 but don't use it, so all
   variants share one launch signature (gs_kernel_t). */
template <int BITS, int THREADS, int CANDS>
__global__ void __launch_bounds__(THREADS, KERNEL_MIN_BLOCKS)
    mfaktc_barrett_gs(unsigned int exp, int96 k_base, const unsigned int *__restrict__ bit_array, unsigned int bits_to_process,
                      int shiftcount, int192 b_preinit, unsigned int *__restrict__ RES, int bit_max64)
{
    int96 f[CANDS], f_base;
    int valid[CANDS];
    int i, total_bit_count;
    unsigned int initial_shifter_value;
    extern __shared__ unsigned short k_deltas[];

    create_k_deltas(bit_array, bits_to_process, &total_bit_count, k_deltas, RES);
    create_fbase96(&f_base, k_base, exp, bits_to_process);
    initial_shifter_value = exp << (32 - shiftcount);

    for (i = threadIdx.x; i < total_bit_count; i += CANDS * blockDim.x) {
#pragma unroll
        for (int c = 0; c < CANDS; c++) {
            // a slot past the last candidate repeats this thread's first one, but doesn't report it
            const int index   = i + c * blockDim.x;
            valid[c]          = index < total_bit_count;
            const int k_delta = k_deltas[valid[c] ? index : i];
            f[c].d0           = __add_cc(f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
            f[c].d1           = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
            f[c].d2           = __addc(f_base.d2, 0);
        }
        gs_test<BITS, CANDS>::run(f, valid, b_preinit, initial_shifter_value, RES, bit_max64);
    }
}

#define GS_VARIANT(BITS, THREADS, CANDS) { mfaktc_barrett_gs<BITS, THREADS, CANDS>, THREADS, CANDS }
#define GS_VARIANTS(BITS)                                                                                                          \
    { GS_VARIANT(BITS, 256, 1), GS_VARIANT(BITS, 128, 1), GS_VARIANT(BITS, 256, 2), GS_VARIANT(BITS, 128, 2) }

static const gs_variant_t gs_variants_barrett92[] = GS_VARIANTS(92);
static const gs_variant_t gs_variants_barrett88[] = GS_VARIANTS(88);
static const gs_variant_t gs_variants_barrett87[] = GS_VARIANTS(87);
static const gs_variant_t gs_variants_barrett79[] = GS_VARIANTS(79);
static const gs_variant_t gs_variants_barrett77[] = GS_VARIANTS(77);
static const gs_variant_t gs_variants_barrett76[] = GS_VARIANTS(76);
#undef GS_VARIANTS
#undef GS_VARIANT
#endif

#define TF_BARRETT

#define TF_BARRETT_92BIT_GS
#include "tf_common_gs.cu"
#undef TF_BARRETT_92BIT_GS

#define TF_BARRETT_88BIT_GS
#include "tf_common_gs.cu"
#undef TF_BARRETT_88BIT_GS

#define TF_BARRETT_87BIT_GS
#include "tf_common_gs.cu"
#undef TF_BARRETT_87BIT_GS

#define TF_BARRETT_79BIT_GS
#include "tf_common_gs.cu"
#undef TF_BARRETT_79BIT_GS

#define TF_BARRETT_77BIT_GS
#include "tf_common_gs.cu"
#undef TF_BARRETT_77BIT_GS

#define TF_BARRETT_76BIT_GS
#include "tf_common_gs.cu"
#undef TF_BARRETT_76BIT_GS

#undef TF_BARRETT
