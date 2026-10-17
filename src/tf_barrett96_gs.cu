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

/* gs_test<BITS, CANDS, BIT_MAX64>::run() tests CANDS candidates with the
   BITS-bit Barrett code.  The generic one-candidate kernels use the original
   test_FC96_barrettXX(); all others use the force-inlined
   test_FC96_barrettXX_n(), so that a compile-time bit_max64 is folded into the
   shifts and constants of the arithmetic. */
template <int BITS, int CANDS, int BIT_MAX64> struct gs_test;

#define GS_BIT_MAX_ARG , bit_max64
#define DEFINE_GS_TEST(BITS, BIT_MAX_ARG)                                                                                          \
    template <int CANDS, int BIT_MAX64> struct gs_test<BITS, CANDS, BIT_MAX64> {                                                   \
        __device__ static __forceinline__ void run(const int96 *f, const int *valid, int192 b, unsigned int shifter, unsigned int *RES, \
                                                   int bit_max64)                                                                  \
        {                                                                                                                          \
            test_FC96_barrett##BITS##_n<CANDS>(f, valid, b, shifter, RES BIT_MAX_ARG);                                             \
        }                                                                                                                          \
    };                                                                                                                             \
    template <> struct gs_test<BITS, 1, 0> {                                                                                       \
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
   the cost of more registers.  WARP_LOCAL kernels take their candidates
   straight from the sieve words held by each warp instead of a block-wide
   list in shared memory; they are launched without dynamic shared memory.
   BIT_MAX64 != 0 compiles a kernel for that one factor size (bit_max64 is
   then ignored).  The host times the variants and keeps the fastest.  The
   76/77/79-bit variants accept bit_max64 but don't use it, so all variants
   share one launch signature (gs_kernel_t). */
template <int BITS, int THREADS, int CANDS, int WARP_LOCAL, int BIT_MAX64>
__global__ void __launch_bounds__(THREADS, KERNEL_MIN_BLOCKS)
    mfaktc_barrett_gs(unsigned int exp, int96 k_base, const unsigned int *__restrict__ bit_array, unsigned int bits_to_process,
                      int shiftcount, int192 b_preinit, unsigned int *__restrict__ RES, int bit_max64)
{
    int96 f[CANDS], f_base;
    int valid[CANDS];
    unsigned int initial_shifter_value;

    if (BIT_MAX64) bit_max64 = BIT_MAX64;

    if (WARP_LOCAL) {
        const int MAXW = GPU_SIEVE_PROCESS_SIZE_MAX * 1024 / (32 * THREADS);
        unsigned int w[MAXW], excl, total, first_word;
        const unsigned int lane = threadIdx.x & (warpSize - 1);

        warp_load_sieve_words<MAXW>(bit_array, bits_to_process, w, &excl, &total, &first_word);
        create_fbase96(&f_base, k_base, exp, bits_to_process);
        initial_shifter_value = exp << (32 - shiftcount);

        // total is the same in all lanes, so the whole warp runs every round
        for (unsigned int base = 0; base < total; base += CANDS * warpSize) {
#pragma unroll
            for (int c = 0; c < CANDS; c++) {
                // ranks past the slice's last candidate repeat the round's first one, but don't report it
                const unsigned int rank = base + c * warpSize + lane;
                valid[c]                = rank < total;
                const int k_delta       = warp_candidate_offset<MAXW>(w, excl, first_word, valid[c] ? rank : base);
                f[c].d0                 = __add_cc(f_base.d0, __umul32(2 * k_delta * NUM_CLASSES, exp));
                f[c].d1                 = __addc_cc(f_base.d1, __umul32hi(2 * k_delta * NUM_CLASSES, exp));
                f[c].d2                 = __addc(f_base.d2, 0);
            }
            gs_test<BITS, CANDS, BIT_MAX64>::run(f, valid, b_preinit, initial_shifter_value, RES, bit_max64);
        }
    } else {
        int i, total_bit_count;
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
            gs_test<BITS, CANDS, BIT_MAX64>::run(f, valid, b_preinit, initial_shifter_value, RES, bit_max64);
        }
    }
}

#define GS_VARIANT(BITS, THREADS, CANDS, WARP_LOCAL, BIT_MAX64)                                                                    \
    { mfaktc_barrett_gs<BITS, THREADS, CANDS, WARP_LOCAL, BIT_MAX64>, THREADS, CANDS, WARP_LOCAL, BIT_MAX64 }
#define GS_GENERIC_VARIANTS(BITS)                                                                                                  \
    GS_VARIANT(BITS, 256, 1, 0, 0), GS_VARIANT(BITS, 128, 1, 0, 0), GS_VARIANT(BITS, 256, 2, 0, 0), GS_VARIANT(BITS, 128, 2, 0, 0), \
        GS_VARIANT(BITS, 256, 1, 1, 0), GS_VARIANT(BITS, 128, 1, 1, 0), GS_VARIANT(BITS, 256, 2, 1, 0), GS_VARIANT(BITS, 128, 2, 1, 0)
#define GS_BIT_LEVEL_VARIANTS(BITS, BIT_MAX64) GS_VARIANT(BITS, 256, 1, 0, BIT_MAX64), GS_VARIANT(BITS, 128, 1, 0, BIT_MAX64)

/* Bit-level kernels exist for the factor sizes the automatic kernel selection
   most often gives each of the 87-, 88- and 92-bit kernels: 2^77 to 2^84,
   2^88 and 2^89 to 2^92.  Other sizes use the generic variants. */
static const gs_variant_t gs_variants_barrett92[] = {
    GS_GENERIC_VARIANTS(92),       GS_BIT_LEVEL_VARIANTS(92, 25), GS_BIT_LEVEL_VARIANTS(92, 26),
    GS_BIT_LEVEL_VARIANTS(92, 27), GS_BIT_LEVEL_VARIANTS(92, 28),
};
static const gs_variant_t gs_variants_barrett88[] = { GS_GENERIC_VARIANTS(88), GS_BIT_LEVEL_VARIANTS(88, 24) };
static const gs_variant_t gs_variants_barrett87[] = {
    GS_GENERIC_VARIANTS(87),       GS_BIT_LEVEL_VARIANTS(87, 13), GS_BIT_LEVEL_VARIANTS(87, 14), GS_BIT_LEVEL_VARIANTS(87, 15),
    GS_BIT_LEVEL_VARIANTS(87, 16), GS_BIT_LEVEL_VARIANTS(87, 17), GS_BIT_LEVEL_VARIANTS(87, 18), GS_BIT_LEVEL_VARIANTS(87, 19),
    GS_BIT_LEVEL_VARIANTS(87, 20),
};
static const gs_variant_t gs_variants_barrett79[] = { GS_GENERIC_VARIANTS(79) };
static const gs_variant_t gs_variants_barrett77[] = { GS_GENERIC_VARIANTS(77) };
static const gs_variant_t gs_variants_barrett76[] = { GS_GENERIC_VARIANTS(76) };
#undef GS_BIT_LEVEL_VARIANTS
#undef GS_GENERIC_VARIANTS
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
