/*
This file is part of mfaktc.
Copyright (C) 2009, 2010, 2011, 2012, 2014  Oliver Weihe (o.weihe@t-online.de)

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

__device__ static void create_k_deltas(const unsigned int *__restrict__ bit_array, unsigned int bits_to_process, int *total_bit_count,
                                       unsigned short *k_deltas)
{
    int i, words_per_thread, sieve_word, k_bit_base;
    unsigned int local_bit_count, inclusive_bit_count;
    const unsigned int lane = threadIdx.x & (warpSize - 1);
    const unsigned int warp = threadIdx.x / warpSize;
    __shared__ unsigned short warp_bitcount[THREADS_PER_BLOCK / 32];

    // Get pointer to section of the bit_array this thread is processing.

    words_per_thread = bits_to_process / (blockDim.x * 32);
    bit_array += blockIdx.x * bits_to_process / 32 + threadIdx.x * words_per_thread;

    // Count number of bits set in this thread's word(s) from the bit_array

    local_bit_count = 0;
    for (i = 0; i < words_per_thread; i++)
        local_bit_count += __popc(bit_array[i]);

    // Inclusive scan within each warp.  Shuffle operations avoid the five shared-memory
    // round trips and four block barriers used by the original 256-entry scan.
    inclusive_bit_count = local_bit_count;
    for (unsigned int offset = 1; offset < warpSize; offset <<= 1) {
        unsigned int value = __shfl_up_sync(0xFFFFFFFFU, inclusive_bit_count, offset);
        if (lane >= offset) inclusive_bit_count += value;
    }

    if (lane == warpSize - 1) warp_bitcount[warp] = inclusive_bit_count;
    __syncthreads();

    // Warp zero scans the per-warp totals.  THREADS_PER_BLOCK is the maximum
    // supported block size; launches may use fewer threads.
    if (warp == 0) {
        unsigned int value = lane < (blockDim.x / warpSize) ? warp_bitcount[lane] : 0;
        for (unsigned int offset = 1; offset < warpSize; offset <<= 1) {
            unsigned int previous = __shfl_up_sync(0xFFFFFFFFU, value, offset);
            if (lane >= offset) value += previous;
        }
        if (lane < (blockDim.x / warpSize)) warp_bitcount[lane] = value;
    }
    __syncthreads();

    unsigned int warp_offset = warp == 0 ? 0 : warp_bitcount[warp - 1];
    inclusive_bit_count += warp_offset;
    *total_bit_count = warp_bitcount[blockDim.x / warpSize - 1];

    //POSSIBLE SANITY CHECK -- is there any way to test if total_bit_count exceeds the amount of shared memory allocated?

    // Loop til this thread's section of the bit array is finished.

    sieve_word = *bit_array;
    k_bit_base = threadIdx.x * words_per_thread * 32;
    for (i = inclusive_bit_count - local_bit_count;; i++) {
        int bit_to_test;

        // Make sure we have a non-zero sieve word

        while (sieve_word == 0) {
            if (--words_per_thread == 0) break;
            sieve_word = *++bit_array;
            k_bit_base += 32;
        }

        // Check if this thread has processed all its set bits

        if (sieve_word == 0) break;

        // Find a bit to test in the sieve word

        bit_to_test = 31 - __clz(sieve_word);
        sieve_word &= ~(1 << bit_to_test);

        // Copy the k value to the shared memory array

        k_deltas[i] = k_bit_base + bit_to_test;
    }

    __syncthreads();
    // Here, all warps in our block have placed their candidates in shared memory.
    // Now we can start TFing candidates.
}

__device__ static void create_fbase96(int96 *f_base, int96 k_base, unsigned int exp, unsigned int bits_to_process)
{
    // Compute factor corresponding to first sieve bit in this block.

    // Compute base k value
    k_base.d0 = __add_cc(k_base.d0, __umul32(blockIdx.x * bits_to_process, NUM_CLASSES));
    k_base.d1 = __addc(k_base.d1, __umul32hi(blockIdx.x * bits_to_process, NUM_CLASSES)); /* k values are limited to 64 bits */

    // Compute k * exp
    f_base->d0 = __umul32(k_base.d0, exp);
    f_base->d1 = __add_cc(__umul32hi(k_base.d0, exp), __umul32(k_base.d1, exp));
    f_base->d2 = __addc(__umul32hi(k_base.d1, exp), 0);

    // Compute f_base = 2 * k * exp + 1
    shl_96(f_base);
    f_base->d0 = f_base->d0 + 1;
}
