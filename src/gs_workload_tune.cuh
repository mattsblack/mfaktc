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

#ifndef MFAKTC_GS_WORKLOAD_TUNE_CUH
#define MFAKTC_GS_WORKLOAD_TUNE_CUH

#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

#include "params.h"
#include "my_types.h"
#include "tf_autotune.cuh"

/* Cached winner of a joint sieve-plus-TF search.  The key is GPU + toolkit +
   kernel family + bit level + exponent category; the rest is the configuration
   to restore.  tf_threads == 0 means the occupancy default (no variant table). */
typedef struct {
    int primes;
    int process_size; /* bits */
    int segment_size; /* bits */
    int atomic_marks;
    int batch_size; /* bits */
    int tf_threads;
    int tf_cands;
    int tf_warp_local;
    int tf_bit_max64;
} gs_tune_cfg_t;

static int gs_exp_category(unsigned int exp)
{
    int n = 0;

    while (exp > 1) {
        exp >>= 1;
        n++;
    }
    return n;
}

static void gs_tune_gpu_name(char *out, size_t out_len)
{
    struct cudaDeviceProp prop;
    size_t i;

    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        snprintf(out, out_len, "unknown");
        return;
    }
    for (i = 0; prop.name[i] && i + 1 < out_len; i++)
        out[i] = (prop.name[i] == ' ' || prop.name[i] == '\t') ? '_' : prop.name[i];
    out[i] = '\0';
}

static int gs_tune_find_variant(const gs_variant_t *variants, int count, const gs_tune_cfg_t *cfg)
{
    for (int v = 0; v < count; v++)
        if (variants[v].threads == cfg->tf_threads && variants[v].cands == cfg->tf_cands &&
            variants[v].warp_local == cfg->tf_warp_local && variants[v].bit_max64 == cfg->tf_bit_max64)
            return v;
    return -1;
}

static void gs_tune_from_variant(gs_tune_cfg_t *cfg, const gs_variant_t *variant)
{
    cfg->tf_threads     = variant->threads;
    cfg->tf_cands       = variant->cands;
    cfg->tf_warp_local  = variant->warp_local;
    cfg->tf_bit_max64   = variant->bit_max64;
}

/* Returns 1 if a matching cache line was found. */
static int gs_tune_cache_load(const char *family, int bit_min, unsigned int exponent, gs_tune_cfg_t *cfg)
{
    FILE *in;
    char gpu[256], line[512], file_ver[32], file_gpu[256], file_family[32];
    int toolkit, file_bit, file_cat, loaded = 0;

    gs_tune_gpu_name(gpu, sizeof(gpu));
    in = fopen(GS_AUTOTUNE_CACHE_FILE, "r");
    if (!in) return 0;

    while (fgets(line, sizeof(line), in)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        if (sscanf(line, "%31s %d %255s %31s %d %d %d %d %d %d %d %d %d %d %d", file_ver, &toolkit, file_gpu, file_family, &file_bit,
                   &file_cat, &cfg->primes, &cfg->process_size, &cfg->segment_size, &cfg->atomic_marks, &cfg->batch_size,
                   &cfg->tf_threads, &cfg->tf_cands, &cfg->tf_warp_local, &cfg->tf_bit_max64) != 15)
            continue;
        if (strcmp(file_ver, MFAKTC_VERSION) == 0 && toolkit == CUDART_VERSION && strcmp(file_gpu, gpu) == 0 &&
            strcmp(file_family, family) == 0 && file_bit == bit_min && file_cat == gs_exp_category(exponent))
            loaded = 1;
    }
    fclose(in);
    return loaded;
}

static void gs_tune_cache_save(const char *family, int bit_min, unsigned int exponent, const gs_tune_cfg_t *cfg)
{
    FILE *in, *out;
    char gpu[256], line[512], tmpname[] = GS_AUTOTUNE_CACHE_FILE ".tmp";
    char file_ver[32], file_gpu[256], file_family[32];
    int toolkit, file_bit, file_cat;
    int dummy[11];

    gs_tune_gpu_name(gpu, sizeof(gpu));
    in  = fopen(GS_AUTOTUNE_CACHE_FILE, "r");
    out = fopen(tmpname, "w");
    if (!out) {
        if (in) fclose(in);
        return;
    }
    fprintf(out, "# mfaktc GPU-sieve autotune cache.  Lines are replaced when the same key is retuned.\n");
    fprintf(out, "# version toolkit gpu family bit_min exp_cat primes process_bits segment_bits atomic batch_bits threads cands "
                 "warp_local bit_max64\n");
    if (in) {
        while (fgets(line, sizeof(line), in)) {
            if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
            if (sscanf(line, "%31s %d %255s %31s %d %d %d %d %d %d %d %d %d %d %d", file_ver, &toolkit, file_gpu, file_family, &file_bit,
                       &file_cat, &dummy[0], &dummy[1], &dummy[2], &dummy[3], &dummy[4], &dummy[5], &dummy[6], &dummy[7],
                       &dummy[8]) == 15 &&
                strcmp(file_ver, MFAKTC_VERSION) == 0 && toolkit == CUDART_VERSION && strcmp(file_gpu, gpu) == 0 &&
                strcmp(file_family, family) == 0 && file_bit == bit_min && file_cat == gs_exp_category(exponent))
                continue;
            fputs(line, out);
        }
        fclose(in);
    }
    fprintf(out, "%s %d %s %s %d %d %d %d %d %d %d %d %d %d %d\n", MFAKTC_VERSION, CUDART_VERSION, gpu, family, bit_min,
            gs_exp_category(exponent), cfg->primes, cfg->process_size, cfg->segment_size, cfg->atomic_marks, cfg->batch_size,
            cfg->tf_threads, cfg->tf_cands, cfg->tf_warp_local, cfg->tf_bit_max64);
    fclose(out);
#ifdef _MSC_VER
    remove(GS_AUTOTUNE_CACHE_FILE);
#endif
    rename(tmpname, GS_AUTOTUNE_CACHE_FILE);
}

/* Shared memory for the k_deltas[] candidate list, matching tf_common_gs.cu. */
static int gs_candidate_shared_mem(const mystuff_t *mystuff, int conservative)
{
    int pct;

#ifdef RAW_GPU_BENCH
    pct = 100;
#else
    if (mystuff->gpu_sieve_primes < 54)
        pct = 100;
    else if (mystuff->gpu_sieve_primes < 310)
        pct = 50;
    else if (mystuff->gpu_sieve_primes < 1846)
        pct = 38;
    else if (mystuff->gpu_sieve_primes < 21814)
        pct = 30;
    else if (mystuff->gpu_sieve_primes < 67894)
        pct = 24;
    else
        pct = 22;
#endif
    {
        const size_t estimated_candidates = ((size_t)mystuff->gpu_sieve_processing_size * (size_t)pct + 99) / 100;
        size_t candidate_capacity         = conservative ? estimated_candidates * 2 : (estimated_candidates * 3 + 1) / 2;
        if (candidate_capacity > (size_t)mystuff->gpu_sieve_processing_size)
            candidate_capacity = (size_t)mystuff->gpu_sieve_processing_size;
        return (int)(((candidate_capacity * sizeof(unsigned short)) + 255) & ~(size_t)255);
    }
}

#endif
