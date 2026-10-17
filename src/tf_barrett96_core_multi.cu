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

/*
N-candidate versions of the GPU-sieve Barrett functions in tf_barrett96_core.cu.
Each test_FC96_barrettXX_n<N>() performs exactly the arithmetic of
test_FC96_barrettXX() for N factor candidates at once. All candidates of a
thread share the exponent, so they also share the shifter and run the main loop
in lockstep. Every step is done for all candidates before the next step starts,
which gives the scheduler N independent dependency chains to interleave.

The carry flag is not visible to the compiler, so a carry chain (the asm
volatile __add_cc() ... __addc() sequences) must never be split by another
candidate's chain: each loop over c below contains complete chains only.

Only candidates with valid[c] != 0 may report a factor; the others are padding
for a partially filled final iteration.

These functions are not used with CPU_SIEVE or DEBUG_GPU_MATH.
*/

template <int N> __device__ static __forceinline__ void barrett96_n_init_ff(float *ff, const int96 *f)
{
#pragma unroll
    for (int c = 0; c < N; c++) {
        ff[c] = __uint2float_rn(f[c].d2);
        ff[c] = ff[c] * 4294967296.0f + __uint2float_rn(f[c].d1);
        ff[c] = __int_as_float(0x3f7ffffb) / ff[c];
    }
}

template <int N>
__device__ static __forceinline__ void barrett96_n_check(const int96 *a, const int96 *f, const float *ff, const int *valid, unsigned int *RES)
{
#pragma unroll
    for (int c = 0; c < N; c++)
        if (valid[c]) mod_simple_96_and_check_big_factor96(a[c], f[c], ff[c], RES);
}

template <int N>
__device__ static __forceinline__ void test_FC96_barrett92_n(const int96 *f, const int *valid, int192 b_preinit, unsigned int shifter, unsigned int *RES,
                                             int bit_max64)
{
    int96 a[N], u[N], tmp96[N];
    int192 b[N], tmp192[N];
    float ff[N];

    barrett96_n_init_ff<N>(ff, f);

#pragma unroll
    for (int c = 0; c < N; c++) {
        tmp192[c].d5 = 1 << (bit_max64 - 1); // tmp192 = 2^(95 + bits_in_f)
        tmp192[c].d4 = 0;
        tmp192[c].d3 = 0;
        tmp192[c].d2 = 0;
        tmp192[c].d1 = 0;
        tmp192[c].d0 = 0;
        div_192_96(&u[c], tmp192[c], f[c], ff[c]); // u = floor(2^(95 + bits_in_f) / f)
    }

#pragma unroll
    for (int c = 0; c < N; c++) {
        a[c].d0 = __fshift_r(b_preinit.d2, b_preinit.d3, bit_max64 - 1); // a = floor(b / 2 ^ (bits_in_f - 1))
        a[c].d1 = __fshift_r(b_preinit.d3, b_preinit.d4, bit_max64 - 1);
        a[c].d2 = __fshift_r(b_preinit.d4, b_preinit.d5, bit_max64 - 1);
        mul_96_192_no_low3(&tmp192[c], a[c], u[c]);
    }

#pragma unroll
    for (int c = 0; c < N; c++) {
        a[c].d0 = tmp192[c].d3; // a = quotient b / f
        a[c].d1 = tmp192[c].d4;
        a[c].d2 = tmp192[c].d5;
        mul_96(&tmp96[c], a[c], f[c]); // low 96 bits of quotient * f
    }

#pragma unroll
    for (int c = 0; c < N; c++) {
        tmp96[c].d0 = __sub_cc(b_preinit.d0, tmp96[c].d0); // remainder
        tmp96[c].d1 = __subc_cc(b_preinit.d1, tmp96[c].d1);
        tmp96[c].d2 = __subc(b_preinit.d2, tmp96[c].d2);
    }

    while (shifter) {
#pragma unroll
        for (int c = 0; c < N; c++)
            mod_simple_96(&a[c], tmp96[c], f[c], ff[c]); // a can be 93 bits

#pragma unroll
        for (int c = 0; c < N; c++)
            square_96_192(&b[c], a[c]); // b = a^2

#pragma unroll
        for (int c = 0; c < N; c++) {
            a[c].d0 = __fshift_r(b[c].d2, b[c].d3, bit_max64 - 1); // a = b / (2 ^ (bits_in_f - 1))
            a[c].d1 = __fshift_r(b[c].d3, b[c].d4, bit_max64 - 1);
            a[c].d2 = __fshift_r(b[c].d4, b[c].d5, bit_max64 - 1);
            mul_96_192_no_low3(&tmp192[c], a[c], u[c]);
        }

#pragma unroll
        for (int c = 0; c < N; c++) {
            a[c].d0 = tmp192[c].d3; // a = quotient b / f, off by at most 6
            a[c].d1 = tmp192[c].d4;
            a[c].d2 = tmp192[c].d5;
            mul_96(&tmp96[c], a[c], f[c]);
        }

#pragma unroll
        for (int c = 0; c < N; c++) {
            tmp96[c].d0 = __sub_cc(b[c].d0, tmp96[c].d0); // remainder, up to 7 * f
            tmp96[c].d1 = __subc_cc(b[c].d1, tmp96[c].d1);
            tmp96[c].d2 = __subc(b[c].d2, tmp96[c].d2);
        }

        if (shifter & 0x80000000) {
#pragma unroll
            for (int c = 0; c < N; c++)
                shl_96(&tmp96[c]);
        }

        shifter += shifter;
    }

    barrett96_n_check<N>(tmp96, f, ff, valid, RES);
}

template <int N>
__device__ static __forceinline__ void test_FC96_barrett88_n(const int96 *f, const int *valid, int192 b_preinit, unsigned int shifter, unsigned int *RES,
                                             int bit_max64)
{
    int96 a[N], u[N], tmp96[N];
    int192 b[N], tmp192[N];
    float ff[N];

    barrett96_n_init_ff<N>(ff, f);

#pragma unroll
    for (int c = 0; c < N; c++) {
        tmp192[c].d5 = 1 << (bit_max64 - 1); // tmp192 = 2^(95 + bits_in_f)
        tmp192[c].d4 = 0;
        tmp192[c].d3 = 0;
        tmp192[c].d2 = 0;
        tmp192[c].d1 = 0;
        tmp192[c].d0 = 0;
        div_192_96(&u[c], tmp192[c], f[c], ff[c]); // u = floor(2^(95 + bits_in_f) / f)
    }

#pragma unroll
    for (int c = 0; c < N; c++) {
        a[c].d0 = __fshift_r(b_preinit.d2, b_preinit.d3, bit_max64 - 1); // a = floor(b / 2 ^ (bits_in_f - 1))
        a[c].d1 = __fshift_r(b_preinit.d3, b_preinit.d4, bit_max64 - 1);
        a[c].d2 = __fshift_r(b_preinit.d4, b_preinit.d5, bit_max64 - 1);
        mul_96_192_no_low3(&tmp192[c], a[c], u[c]);
    }

#pragma unroll
    for (int c = 0; c < N; c++) {
        a[c].d0 = tmp192[c].d3; // a = quotient b / f
        a[c].d1 = tmp192[c].d4;
        a[c].d2 = tmp192[c].d5;
        mul_96(&tmp96[c], a[c], f[c]); // low 96 bits of quotient * f
    }

#pragma unroll
    for (int c = 0; c < N; c++) {
        a[c].d0 = __sub_cc(b_preinit.d0, tmp96[c].d0); // remainder
        a[c].d1 = __subc_cc(b_preinit.d1, tmp96[c].d1);
        a[c].d2 = __subc(b_preinit.d2, tmp96[c].d2);
    }

    while (shifter) {
#pragma unroll
        for (int c = 0; c < N; c++)
            square_96_192(&b[c], a[c]); // b = a^2, at most 181.614 bits

        if (shifter & 0x80000000) {
#pragma unroll
            for (int c = 0; c < N; c++)
                shl_192(&b[c]);
        }

#pragma unroll
        for (int c = 0; c < N; c++) {
            a[c].d0 = __fshift_r(b[c].d2, b[c].d3, bit_max64 - 1); // a = b / (2 ^ (bits_in_f - 1))
            a[c].d1 = __fshift_r(b[c].d3, b[c].d4, bit_max64 - 1);
            a[c].d2 = __fshift_r(b[c].d4, b[c].d5, bit_max64 - 1);
            mul_96_192_no_low3(&tmp192[c], a[c], u[c]);
        }

#pragma unroll
        for (int c = 0; c < N; c++) {
            a[c].d0 = tmp192[c].d3; // a = quotient b / f, off by at most 6
            a[c].d1 = tmp192[c].d4;
            a[c].d2 = tmp192[c].d5;
            mul_96(&tmp96[c], a[c], f[c]);
        }

#pragma unroll
        for (int c = 0; c < N; c++) {
            a[c].d0 = __sub_cc(b[c].d0, tmp96[c].d0); // remainder, up to 7 * f
            a[c].d1 = __subc_cc(b[c].d1, tmp96[c].d1);
            a[c].d2 = __subc(b[c].d2, tmp96[c].d2);
        }

        shifter += shifter;
    }

    barrett96_n_check<N>(a, f, ff, valid, RES);
}

template <int N>
__device__ static __forceinline__ void test_FC96_barrett87_n(const int96 *f, const int *valid, int192 b_preinit, unsigned int shifter, unsigned int *RES,
                                             int bit_max64)
{
    int96 a[N], u[N], tmp96[N];
    int192 b[N], tmp192[N];
    float ff[N];

    barrett96_n_init_ff<N>(ff, f);

#pragma unroll
    for (int c = 0; c < N; c++) {
        tmp192[c].d5 = 1 << (bit_max64 - 1); // tmp192 = 2^(95 + bits_in_f)
        tmp192[c].d4 = 0;
        tmp192[c].d3 = 0;
        tmp192[c].d2 = 0;
        tmp192[c].d1 = 0;
        tmp192[c].d0 = 0;
        div_192_96(&u[c], tmp192[c], f[c], ff[c]); // u = floor(2^(95 + bits_in_f) / f)
    }

#pragma unroll
    for (int c = 0; c < N; c++) {
        a[c].d0 = __fshift_r(b_preinit.d2, b_preinit.d3, bit_max64 - 1); // a = floor(b / 2 ^ (bits_in_f - 1))
        a[c].d1 = __fshift_r(b_preinit.d3, b_preinit.d4, bit_max64 - 1);
        a[c].d2 = __fshift_r(b_preinit.d4, b_preinit.d5, bit_max64 - 1);
        mul_96_192_no_low3(&tmp192[c], a[c], u[c]);
    }

#pragma unroll
    for (int c = 0; c < N; c++) {
        a[c].d0 = tmp192[c].d3; // a = quotient b / f
        a[c].d1 = tmp192[c].d4;
        a[c].d2 = tmp192[c].d5;
        mul_96(&tmp96[c], a[c], f[c]); // low 96 bits of quotient * f
    }

#pragma unroll
    for (int c = 0; c < N; c++) {
        a[c].d0 = __sub_cc(b_preinit.d0, tmp96[c].d0); // remainder
        a[c].d1 = __subc_cc(b_preinit.d1, tmp96[c].d1);
        a[c].d2 = __subc(b_preinit.d2, tmp96[c].d2);
    }

    while (shifter) {
#pragma unroll
        for (int c = 0; c < N; c++)
            square_96_192(&b[c], a[c]); // b = a^2, at most 181.614 bits

#pragma unroll
        for (int c = 0; c < N; c++) {
            a[c].d0 = __fshift_r(b[c].d2, b[c].d3, bit_max64 - 1); // a = b / (2 ^ (bits_in_f - 1))
            a[c].d1 = __fshift_r(b[c].d3, b[c].d4, bit_max64 - 1);
            a[c].d2 = __fshift_r(b[c].d4, b[c].d5, bit_max64 - 1);
            mul_96_192_no_low3(&tmp192[c], a[c], u[c]);
        }

#pragma unroll
        for (int c = 0; c < N; c++) {
            a[c].d0 = tmp192[c].d3; // a = quotient b / f, off by at most 6
            a[c].d1 = tmp192[c].d4;
            a[c].d2 = tmp192[c].d5;
            mul_96(&tmp96[c], a[c], f[c]);
        }

#pragma unroll
        for (int c = 0; c < N; c++) {
            a[c].d0 = __sub_cc(b[c].d0, tmp96[c].d0); // remainder, up to 7 * f
            a[c].d1 = __subc_cc(b[c].d1, tmp96[c].d1);
            a[c].d2 = __subc(b[c].d2, tmp96[c].d2);
        }

        if (shifter & 0x80000000) {
#pragma unroll
            for (int c = 0; c < N; c++)
                shl_96(&a[c]);
        }

        shifter += shifter;
    }

    barrett96_n_check<N>(a, f, ff, valid, RES);
}

template <int N>
__device__ static __forceinline__ void test_FC96_barrett79_n(const int96 *f, const int *valid, int192 b_preinit, unsigned int shifter, unsigned int *RES)
{
    int96 a[N], u[N], tmp96[N];
    int192 b[N], tmp192[N];
    float ff[N];

    barrett96_n_init_ff<N>(ff, f);

#pragma unroll
    for (int c = 0; c < N; c++)
        inv_160_96(&u[c], f[c], ff[c]); // u = floor(2^160 / f)

#pragma unroll
    for (int c = 0; c < N; c++) {
        a[c].d0 = b_preinit.d2; // a = floor(b / 2^64)
        a[c].d1 = b_preinit.d3;
        a[c].d2 = b_preinit.d4;
        mul_96_192_no_low3(&tmp192[c], a[c], u[c]);
    }

#pragma unroll
    for (int c = 0; c < N; c++) {
        a[c].d0 = tmp192[c].d3; // a = quotient b / f
        a[c].d1 = tmp192[c].d4;
        a[c].d2 = tmp192[c].d5;
        mul_96(&tmp96[c], a[c], f[c]); // low 96 bits of quotient * f
    }

#pragma unroll
    for (int c = 0; c < N; c++) {
        tmp96[c].d0 = __sub_cc(b_preinit.d0, tmp96[c].d0); // remainder
        tmp96[c].d1 = __subc_cc(b_preinit.d1, tmp96[c].d1);
        tmp96[c].d2 = __subc(b_preinit.d2, tmp96[c].d2);
    }

    while (shifter) {
#pragma unroll
        for (int c = 0; c < N; c++)
            mod_simple_96(&a[c], tmp96[c], f[c], ff[c]); // a can be 80 bits

#pragma unroll
        for (int c = 0; c < N; c++)
            square_96_160(&b[c], a[c]); // b = a^2, at most 158 bits

#pragma unroll
        for (int c = 0; c < N; c++) {
            a[c].d0 = b[c].d2; // a = floor(b / 2^64)
            a[c].d1 = b[c].d3;
            a[c].d2 = b[c].d4;
            mul_96_192_no_low3(&tmp192[c], a[c], u[c]);
        }

#pragma unroll
        for (int c = 0; c < N; c++) {
            a[c].d0 = tmp192[c].d3; // a = quotient b / f, off by at most 5
            a[c].d1 = tmp192[c].d4;
            a[c].d2 = tmp192[c].d5;
            mul_96(&tmp96[c], a[c], f[c]);
        }

#pragma unroll
        for (int c = 0; c < N; c++) {
            tmp96[c].d0 = __sub_cc(b[c].d0, tmp96[c].d0); // remainder, up to 6 * f
            tmp96[c].d1 = __subc_cc(b[c].d1, tmp96[c].d1);
            tmp96[c].d2 = __subc(b[c].d2, tmp96[c].d2);
        }

        if (shifter & 0x80000000) {
#pragma unroll
            for (int c = 0; c < N; c++)
                shl_96(&tmp96[c]);
        }

        shifter += shifter;
    }

    barrett96_n_check<N>(tmp96, f, ff, valid, RES);
}

template <int N>
__device__ static __forceinline__ void test_FC96_barrett77_n(const int96 *f, const int *valid, int192 b_preinit, unsigned int shifter, unsigned int *RES)
{
    int96 a[N], u[N], tmp96[N];
    int192 b[N], tmp192[N];
    float ff[N];

    barrett96_n_init_ff<N>(ff, f);

#pragma unroll
    for (int c = 0; c < N; c++)
        inv_160_96(&u[c], f[c], ff[c]); // u = floor(2^160 / f)

#pragma unroll
    for (int c = 0; c < N; c++) {
        a[c].d0 = b_preinit.d2; // a = floor(b / 2^64)
        a[c].d1 = b_preinit.d3;
        a[c].d2 = b_preinit.d4;
        mul_96_192_no_low3(&tmp192[c], a[c], u[c]);
    }

#pragma unroll
    for (int c = 0; c < N; c++) {
        a[c].d0 = tmp192[c].d3; // a = quotient b / f
        a[c].d1 = tmp192[c].d4;
        a[c].d2 = tmp192[c].d5;
        mul_96(&tmp96[c], a[c], f[c]); // low 96 bits of quotient * f
    }

#pragma unroll
    for (int c = 0; c < N; c++) {
        a[c].d0 = __sub_cc(b_preinit.d0, tmp96[c].d0); // remainder
        a[c].d1 = __subc_cc(b_preinit.d1, tmp96[c].d1);
        a[c].d2 = __subc(b_preinit.d2, tmp96[c].d2);
    }

    while (shifter) {
#pragma unroll
        for (int c = 0; c < N; c++)
            square_96_160(&b[c], a[c]); // b = a^2, at most 158.644 bits

        if (shifter & 0x80000000) {
#pragma unroll
            for (int c = 0; c < N; c++)
                shl_192(&b[c]);
        }

#pragma unroll
        for (int c = 0; c < N; c++) {
            a[c].d0 = b[c].d2; // a = floor(b / 2^64)
            a[c].d1 = b[c].d3;
            a[c].d2 = b[c].d4;
            mul_96_192_no_low3_special(&tmp192[c], a[c], u[c]);
        }

#pragma unroll
        for (int c = 0; c < N; c++) {
            a[c].d0 = tmp192[c].d3; // a = quotient b / f, off by at most 4
            a[c].d1 = tmp192[c].d4;
            a[c].d2 = tmp192[c].d5;
            mul_96(&tmp96[c], a[c], f[c]);
        }

#pragma unroll
        for (int c = 0; c < N; c++) {
            a[c].d0 = __sub_cc(b[c].d0, tmp96[c].d0); // remainder, up to 5 * f
            a[c].d1 = __subc_cc(b[c].d1, tmp96[c].d1);
            a[c].d2 = __subc(b[c].d2, tmp96[c].d2);
        }

        shifter += shifter;
    }

    barrett96_n_check<N>(a, f, ff, valid, RES);
}

template <int N>
__device__ static __forceinline__ void test_FC96_barrett76_n(const int96 *f, const int *valid, int192 b_preinit, unsigned int shifter, unsigned int *RES)
{
    int96 a[N], u[N], tmp96[N];
    int192 b[N], tmp192[N];
    float ff[N];

    barrett96_n_init_ff<N>(ff, f);

#pragma unroll
    for (int c = 0; c < N; c++)
        inv_160_96(&u[c], f[c], ff[c]); // u = floor(2^160 / f)

#pragma unroll
    for (int c = 0; c < N; c++) {
        a[c].d0 = b_preinit.d2; // a = floor(b / 2^64)
        a[c].d1 = b_preinit.d3;
        a[c].d2 = b_preinit.d4;
        mul_96_192_no_low3(&tmp192[c], a[c], u[c]);
    }

#pragma unroll
    for (int c = 0; c < N; c++) {
        a[c].d0 = tmp192[c].d3; // a = quotient b / f
        a[c].d1 = tmp192[c].d4;
        a[c].d2 = tmp192[c].d5;
        mul_96(&tmp96[c], a[c], f[c]); // low 96 bits of quotient * f
    }

#pragma unroll
    for (int c = 0; c < N; c++) {
        a[c].d0 = __sub_cc(b_preinit.d0, tmp96[c].d0); // remainder
        a[c].d1 = __subc_cc(b_preinit.d1, tmp96[c].d1);
        a[c].d2 = __subc(b_preinit.d2, tmp96[c].d2);
    }

    while (shifter) {
#pragma unroll
        for (int c = 0; c < N; c++)
            square_96_160(&b[c], a[c]); // b = a^2, at most 159.17 bits

#pragma unroll
        for (int c = 0; c < N; c++) {
            a[c].d0 = b[c].d2; // a = floor(b / 2^64)
            a[c].d1 = b[c].d3;
            a[c].d2 = b[c].d4;
            mul_96_192_no_low3(&tmp192[c], a[c], u[c]);
        }

#pragma unroll
        for (int c = 0; c < N; c++) {
            a[c].d0 = tmp192[c].d3; // a = quotient b / f, off by at most 5
            a[c].d1 = tmp192[c].d4;
            a[c].d2 = tmp192[c].d5;
            mul_96(&tmp96[c], a[c], f[c]);
        }

#pragma unroll
        for (int c = 0; c < N; c++) {
            a[c].d0 = __sub_cc(b[c].d0, tmp96[c].d0); // remainder, up to 6 * f
            a[c].d1 = __subc_cc(b[c].d1, tmp96[c].d1);
            a[c].d2 = __subc(b[c].d2, tmp96[c].d2);
        }

        if (shifter & 0x80000000) {
#pragma unroll
            for (int c = 0; c < N; c++)
                shl_96(&a[c]);
        }

        shifter += shifter;
    }

    barrett96_n_check<N>(a, f, ff, valid, RES);
}
