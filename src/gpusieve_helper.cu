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

/* create_k_deltas() stores per-candidate bit offsets in [0, bits_to_process)
   into k_deltas[] as unsigned short. bits_to_process is gpu_sieve_processing_size,
   which is at most GPU_SIEVE_PROCESS_SIZE_MAX * 1024 bits; that must stay within
   the 16-bit range or the offsets would silently truncate. */
static_assert(GPU_SIEVE_PROCESS_SIZE_MAX * 1024 <= 65536, "GPU_SIEVE_PROCESS_SIZE_MAX too large for unsigned short k_deltas[]");

__device__ static void create_k_deltas(const unsigned int *__restrict__ bit_array, unsigned int bits_to_process, int *total_bit_count,
                                       unsigned short *k_deltas, unsigned int *RES)
{
    unsigned int word, bit_count, local_bit_count, inclusive_bit_count, dynamic_smem_size, max_bit_count;
    const unsigned int lane      = threadIdx.x & (warpSize - 1);
    const unsigned int warp      = threadIdx.x / warpSize;
    const unsigned int num_words = bits_to_process / 32;
    __shared__ unsigned int warp_bitcount[THREADS_PER_BLOCK / 32];

    // Get pointer to section of the bit_array this block is processing.  Words are
    // assigned block-strided so adjacent lanes read adjacent words (coalesced loads).
    // This means k_deltas[] is no longer ordered by bit position, which is fine
    // because every candidate is tested independently.

    bit_array += blockIdx.x * num_words;

    // Count number of bits set in this thread's word(s) from the bit_array

    local_bit_count = 0;
    for (word = threadIdx.x; word < num_words; word += blockDim.x)
        local_bit_count += __popc(bit_array[word]);

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
    bit_count = warp_bitcount[blockDim.x / warpSize - 1];

    // the host estimates the size of the dynamic shared memory k_deltas[]
    // from the sieve parameters. If a block has more candidates than its
    // buffer holds, then none are stored or tested. In this case, RES[31] is
    // set so that the host stops instead of silently skipping them. However,
    // this is not expected with the sizes used.
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dynamic_smem_size));
    max_bit_count = dynamic_smem_size / sizeof(unsigned short);
    if (bit_count > max_bit_count) {
        // atomicMax keeps the reported count well-defined when several blocks overflow
        if (threadIdx.x == 0) atomicMax(&RES[31], bit_count);
        *total_bit_count = 0;
        return; // bit_count is block-uniform, so every thread of the block returns here
    }
    *total_bit_count = bit_count;

    // Copy the k values of this thread's words to the shared memory array

    unsigned int i = inclusive_bit_count - local_bit_count;
    for (word = threadIdx.x; word < num_words; word += blockDim.x) {
        unsigned int sieve_word = bit_array[word];
        while (sieve_word != 0) {
            k_deltas[i++] = (word << 5) + (__ffs(sieve_word) - 1);
            sieve_word &= sieve_word - 1; // clear lowest set bit
        }
    }

    __syncthreads();
    // Here, all warps in our block have placed their candidates in shared memory.
    // Now we can start TFing candidates.
}

/*
Warp-local candidate extraction, an alternative to create_k_deltas() that
needs no shared memory and no block-wide synchronization. Each warp owns a
contiguous slice of its block's sieve words and keeps it in registers: lane L
holds words L, L + 32, L + 64, ... of the slice (coalesced loads). The
surviving candidates of the slice are numbered 0 .. total - 1 in lane order,
and warp_candidate_offset() finds the sieve bit of any of them, so the lanes
of a warp can take 32 consecutive candidates per round, keeping the warp full
until the slice's last round.

MAXW is the most words per lane: launches must use the block size the kernel
was compiled for.
*/

// position of the n-th (from 0) set bit of w, which has more than n set bits
__device__ static unsigned int nth_set_bit(unsigned int w, unsigned int n)
{
    unsigned int pos = 0, c;

    c = __popc(w & 0xFFFF);
    if (n >= c) {
        n -= c;
        pos += 16;
        w >>= 16;
    }
    c = __popc(w & 0xFF);
    if (n >= c) {
        n -= c;
        pos += 8;
        w >>= 8;
    }
    c = __popc(w & 0xF);
    if (n >= c) {
        n -= c;
        pos += 4;
        w >>= 4;
    }
    c = __popc(w & 0x3);
    if (n >= c) {
        n -= c;
        pos += 2;
        w >>= 2;
    }
    if (n >= (w & 1)) pos += 1;
    return pos;
}

/* Loads the warp's slice and returns, for this lane, the number of candidates
   in lower lanes (*excl), the slice's candidate count (*total) and the
   slice's first word within the block (*first_word). */
template <int MAXW>
__device__ static void warp_load_sieve_words(const unsigned int *__restrict__ bit_array, unsigned int bits_to_process, unsigned int *w,
                                             unsigned int *excl, unsigned int *total, unsigned int *first_word)
{
    const unsigned int lane           = threadIdx.x & (warpSize - 1);
    const unsigned int warp           = threadIdx.x / warpSize;
    const unsigned int words_per_warp = bits_to_process / 32 / (blockDim.x / warpSize); // a multiple of 32
    const unsigned int words_per_lane = words_per_warp / warpSize;
    unsigned int count                = 0, inclusive;

    *first_word = warp * words_per_warp;
    bit_array += blockIdx.x * (bits_to_process / 32) + *first_word;
#pragma unroll
    for (int j = 0; j < MAXW; j++) {
        w[j] = (unsigned int)j < words_per_lane ? bit_array[j * warpSize + lane] : 0;
        count += __popc(w[j]);
    }

    inclusive = count;
    for (unsigned int offset = 1; offset < warpSize; offset <<= 1) {
        unsigned int value = __shfl_up_sync(0xFFFFFFFFU, inclusive, offset);
        if (lane >= offset) inclusive += value;
    }
    *total = __shfl_sync(0xFFFFFFFFU, inclusive, warpSize - 1);
    *excl  = inclusive - count;
}

/* Returns the bit offset within the block of the slice's candidate number
   rank (< total).  All lanes of the warp must call this together; each may
   ask for a different rank. */
template <int MAXW>
__device__ static unsigned int warp_candidate_offset(const unsigned int *w, unsigned int excl, unsigned int first_word, unsigned int rank)
{
    unsigned int src = 0, n, bits = 0, word = 0;
    int found = 0;

    // the candidate is in the last lane with no more than rank candidates in lower lanes
#pragma unroll
    for (unsigned int step = 16; step > 0; step >>= 1)
        if (__shfl_sync(0xFFFFFFFFU, excl, src + step) <= rank) src += step;

    n = rank - __shfl_sync(0xFFFFFFFFU, excl, src);
#pragma unroll
    for (int j = 0; j < MAXW; j++) {
        const unsigned int x = __shfl_sync(0xFFFFFFFFU, w[j], src);
        const unsigned int c = __popc(x);
        if (!found) {
            if (n < c) {
                bits  = x;
                word  = j;
                found = 1;
            } else
                n -= c;
        }
    }
    return ((first_word + word * warpSize + src) << 5) + nth_set_bit(bits, n);
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
