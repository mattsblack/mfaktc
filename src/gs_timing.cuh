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

#ifndef MFAKTC_GS_TIMING_CUH
#define MFAKTC_GS_TIMING_CUH

#include <cuda_runtime.h>

/* GPU time spent in the sieve and TF kernels of one class.  Each batch records
   CUDA events in one of a few slots; a slot is read back (which waits for its
   batch to finish) only when it is reused or the class ends, so the GPU keeps
   a few batches queued.  Event recording is skipped entirely when disabled. */

#define GS_TIMING_SLOTS 4

enum { GS_EV_SIEVE_START, GS_EV_SIEVE_END, GS_EV_TF_START, GS_EV_TF_END, GS_EV_COUNT };

typedef struct {
    int enabled;
    int created;
    int slot;
    int pending[GS_TIMING_SLOTS];
    cudaEvent_t ev[GS_TIMING_SLOTS][GS_EV_COUNT];
    double sieve_ms, tf_ms;
    int batches;
} gs_timing_t;

static gs_timing_t gs_class_timing;

static cudaError_t gs_timing_begin_class(gs_timing_t *t, int enabled)
{
    cudaError_t err = cudaSuccess;

    t->enabled  = enabled;
    t->slot     = 0;
    t->sieve_ms = 0.0;
    t->tf_ms    = 0.0;
    t->batches  = 0;
    for (int s = 0; s < GS_TIMING_SLOTS; s++)
        t->pending[s] = 0;
    if (enabled && !t->created) {
        for (int s = 0; s < GS_TIMING_SLOTS && err == cudaSuccess; s++)
            for (int e = 0; e < GS_EV_COUNT && err == cudaSuccess; e++)
                err = cudaEventCreate(&t->ev[s][e]);
        if (err == cudaSuccess) t->created = 1;
    }
    return err;
}

static cudaError_t gs_timing_collect(gs_timing_t *t, int s)
{
    cudaError_t err = cudaSuccess;
    float sieve, tf;

    if (!t->pending[s]) return cudaSuccess;
    t->pending[s] = 0;
    err = cudaEventSynchronize(t->ev[s][GS_EV_TF_END]);
    if (err == cudaSuccess) err = cudaEventSynchronize(t->ev[s][GS_EV_SIEVE_END]);
    if (err == cudaSuccess) err = cudaEventElapsedTime(&sieve, t->ev[s][GS_EV_SIEVE_START], t->ev[s][GS_EV_SIEVE_END]);
    if (err == cudaSuccess) err = cudaEventElapsedTime(&tf, t->ev[s][GS_EV_TF_START], t->ev[s][GS_EV_TF_END]);
    if (err == cudaSuccess) {
        t->sieve_ms += sieve;
        t->tf_ms += tf;
        t->batches++;
    }
    return err;
}

/* Call before recording the events of a batch; frees the slot it will use. */
static cudaError_t gs_timing_begin_batch(gs_timing_t *t)
{
    return t->enabled ? gs_timing_collect(t, t->slot) : cudaSuccess;
}

static cudaError_t gs_timing_record(gs_timing_t *t, int event, cudaStream_t stream)
{
    return t->enabled ? cudaEventRecord(t->ev[t->slot][event], stream) : cudaSuccess;
}

static void gs_timing_end_batch(gs_timing_t *t)
{
    if (!t->enabled) return;
    t->pending[t->slot] = 1;
    t->slot             = (t->slot + 1) % GS_TIMING_SLOTS;
}

static cudaError_t gs_timing_end_class(gs_timing_t *t)
{
    cudaError_t err = cudaSuccess;

    for (int s = 0; s < GS_TIMING_SLOTS && err == cudaSuccess; s++)
        err = gs_timing_collect(t, s);
    return err;
}

#endif
