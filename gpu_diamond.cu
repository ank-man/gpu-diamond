/* GPU-Diamond: standalone CUDA implementation of seed-and-extend
 * protein alignment, inspired by GPU-BLAST and Diamond.
 *
 * Layout:
 *   db_padded[seq_idx * padded_len + offset]   (row-major, 1 byte / AA)
 *   matrix[q*28 + s]                            (BLOSUM62 row-major)
 *   bucket_count[hash]                          (#query positions for hash)
 *   bucket_pos[hash * MAX_HITS_BUCKET + i]      (i-th query position)
 *
 * Each CUDA thread processes one or more subject sequences (strided by
 * grid*block size), scans every 4-mer, looks up query hits, performs
 * ungapped extension with X-dropoff, and stores HSPs in DiamondResult.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpu_diamond.h"

/* ---------- Device constants ---------- */
__constant__ signed char d_matrix[DIAMOND_ALPHABET_SIZE * DIAMOND_ALPHABET_SIZE];
__constant__ unsigned char d_query[DIAMOND_MAX_QUERY_LEN];
__constant__ int d_qlen;
__constant__ int d_xdrop;
__constant__ int d_min_score;
__constant__ int d_seed_score_min;

/* ---------- Device helpers ---------- */
__device__ __forceinline__ uint32_t d_hash4(const unsigned char* s)
{
    return ((uint32_t)s[0] * (28u * 28u * 28u))
         + ((uint32_t)s[1] * (28u * 28u))
         + ((uint32_t)s[2] * 28u)
         + ((uint32_t)s[3]);
}

/* Score a 4-mer match using BLOSUM62 */
__device__ __forceinline__ int d_seed_score(const unsigned char* subj,
                                            int q_off)
{
    int sc = 0;
    #pragma unroll
    for (int i = 0; i < DIAMOND_SEED_SIZE; ++i) {
        unsigned char q = d_query[q_off + i];
        unsigned char s = subj[i];
        sc += d_matrix[q * DIAMOND_ALPHABET_SIZE + s];
    }
    return sc;
}

/* Ungapped extension: extend left and right from (s_off, q_off).
 * Returns max score and bounds. */
struct ExtResult { int score, q_lo, q_hi, s_lo, s_hi; };

__device__ ExtResult d_ungapped_extend(const unsigned char* subj, int slen,
                                       int s_off, int q_off,
                                       int x_drop)
{
    /* The 4-mer match itself */
    int seed = d_seed_score(subj + s_off, q_off);

    /* Extend right (after the seed) */
    int score      = seed;
    int max_score  = seed;
    int max_dr     = DIAMOND_SEED_SIZE - 1;   /* index of last accepted position */
    int s = s_off + DIAMOND_SEED_SIZE;
    int q = q_off + DIAMOND_SEED_SIZE;
    int dr = DIAMOND_SEED_SIZE;
    while (s < slen && q < d_qlen) {
        score += d_matrix[d_query[q] * DIAMOND_ALPHABET_SIZE + subj[s]];
        if (score > max_score) { max_score = score; max_dr = dr; }
        else if (max_score - score > x_drop) break;
        ++s; ++q; ++dr;
    }

    /* Extend left (before the seed) starting from the right-extended max */
    int left_score = 0;
    int max_left   = 0;
    int max_dl     = 0;
    int sl = s_off - 1;
    int ql = q_off - 1;
    int dl = 1;
    while (sl >= 0 && ql >= 0) {
        left_score += d_matrix[d_query[ql] * DIAMOND_ALPHABET_SIZE + subj[sl]];
        if (left_score > max_left) { max_left = left_score; max_dl = dl; }
        else if (max_left - left_score > x_drop) break;
        --sl; --ql; ++dl;
    }

    ExtResult r;
    r.score = max_score + max_left;
    r.q_lo  = q_off - max_dl;
    r.q_hi  = q_off + max_dr;     /* inclusive end */
    r.s_lo  = s_off - max_dl;
    r.s_hi  = s_off + max_dr;
    return r;
}

/* ---------- Main kernel ---------- */
__global__ void diamond_kernel(const unsigned char* __restrict__ db_padded,
                               const int*           __restrict__ seq_lengths,
                               int                  num_sequences,
                               int                  padded_len,
                               const uint32_t*      __restrict__ bucket_count,
                               const uint32_t*      __restrict__ bucket_pos,
                               DiamondResult*       __restrict__ results)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    /* Per-thread diagonal table (in registers/local memory).
     * Stores last subject offset extended for diagonal d (mod DIAMOND_DIAG_SIZE).
     * Init to -DIAMOND_SEED_SIZE so any first hit is accepted. */
    int diag_last[DIAMOND_DIAG_SIZE];
    #pragma unroll 1
    for (int i = 0; i < DIAMOND_DIAG_SIZE; ++i) diag_last[i] = -DIAMOND_SEED_SIZE;

    for (int seq = tid; seq < num_sequences; seq += stride) {
        const unsigned char* subj = db_padded + (size_t)seq * padded_len;
        int slen = seq_lengths[seq];

        DiamondResult res;
        res.num_hsps = 0;
        res.total_score = 0;
        res.seed_hits = 0;
        res.extensions = 0;

        /* Reset diag table for this sequence */
        for (int i = 0; i < DIAMOND_DIAG_SIZE; ++i) diag_last[i] = -DIAMOND_SEED_SIZE;

        if (slen >= DIAMOND_SEED_SIZE) {
            for (int s_off = 0; s_off + DIAMOND_SEED_SIZE <= slen; ++s_off) {
                uint32_t h = d_hash4(subj + s_off);
                uint32_t cnt = bucket_count[h];
                if (cnt == 0) continue;

                if (cnt > DIAMOND_MAX_HITS_BUCKET) cnt = DIAMOND_MAX_HITS_BUCKET;
                res.seed_hits += (int)cnt;

                for (uint32_t k = 0; k < cnt; ++k) {
                    int q_off = (int)bucket_pos[h * DIAMOND_MAX_HITS_BUCKET + k];

                    /* Diagonal de-dup */
                    int diag = (q_off - s_off) & DIAMOND_DIAG_MASK;
                    if (s_off - diag_last[diag] < DIAMOND_SEED_SIZE) continue;

                    int sscore = d_seed_score(subj + s_off, q_off);
                    if (sscore < d_seed_score_min) continue;

                    res.extensions++;

                    ExtResult e = d_ungapped_extend(subj, slen, s_off, q_off, d_xdrop);
                    diag_last[diag] = e.s_hi;

                    if (e.score >= d_min_score && res.num_hsps < DIAMOND_MAX_HSPS) {
                        DiamondHSP* hsp = &res.hsps[res.num_hsps++];
                        hsp->q_start = e.q_lo;
                        hsp->q_end   = e.q_hi;
                        hsp->s_start = e.s_lo;
                        hsp->s_end   = e.s_hi;
                        hsp->score   = e.score;
                        res.total_score += e.score;
                    }
                }
            }
        }

        results[seq] = res;
    }
}

/* ---------- Host launcher ---------- */
#define CUDA_CHECK(call) do {                                          \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call,         \
                __FILE__, __LINE__, cudaGetErrorString(_e));           \
        return (int)_e;                                                \
    }                                                                  \
} while (0)

extern "C"
int diamond_gpu_search(const unsigned char* query, int qlen,
                       const unsigned char* db_padded,
                       const int*           seq_lengths,
                       const DiamondGPUOptions* opts,
                       DiamondResult*       results)
{
    if (qlen <= 0 || qlen > DIAMOND_MAX_QUERY_LEN) {
        fprintf(stderr, "Query length %d out of range [1, %d]\n",
                qlen, DIAMOND_MAX_QUERY_LEN);
        return -1;
    }
    if (opts->num_sequences <= 0) return 0;

    /* ----- Build host-side lookup table ----- */
    uint32_t* h_bucket_count = (uint32_t*)calloc(DIAMOND_HASH_SIZE, sizeof(uint32_t));
    uint32_t* h_bucket_pos   = (uint32_t*)calloc((size_t)DIAMOND_HASH_SIZE * DIAMOND_MAX_HITS_BUCKET, sizeof(uint32_t));
    if (!h_bucket_count || !h_bucket_pos) {
        fprintf(stderr, "Out of host memory for lookup table\n");
        free(h_bucket_count); free(h_bucket_pos);
        return -2;
    }
    diamond_build_lookup(query, qlen, h_bucket_count, h_bucket_pos);

    /* ----- BLOSUM62 to constant memory ----- */
    signed char h_matrix[DIAMOND_ALPHABET_SIZE * DIAMOND_ALPHABET_SIZE];
    diamond_fill_blosum62(h_matrix);

    /* ----- Allocate GPU buffers ----- */
    unsigned char* d_db = NULL;
    int*           d_lens = NULL;
    uint32_t*      d_bcount = NULL;
    uint32_t*      d_bpos = NULL;
    DiamondResult* d_results = NULL;

    size_t db_bytes      = (size_t)opts->num_sequences * opts->padded_length;
    size_t lens_bytes    = (size_t)opts->num_sequences * sizeof(int);
    size_t bcount_bytes  = (size_t)DIAMOND_HASH_SIZE * sizeof(uint32_t);
    size_t bpos_bytes    = (size_t)DIAMOND_HASH_SIZE * DIAMOND_MAX_HITS_BUCKET * sizeof(uint32_t);
    size_t results_bytes = (size_t)opts->num_sequences * sizeof(DiamondResult);

    CUDA_CHECK(cudaMalloc(&d_db,      db_bytes));
    CUDA_CHECK(cudaMalloc(&d_lens,    lens_bytes));
    CUDA_CHECK(cudaMalloc(&d_bcount,  bcount_bytes));
    CUDA_CHECK(cudaMalloc(&d_bpos,    bpos_bytes));
    CUDA_CHECK(cudaMalloc(&d_results, results_bytes));

    CUDA_CHECK(cudaMemcpy(d_db,     db_padded,      db_bytes,     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lens,   seq_lengths,    lens_bytes,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bcount, h_bucket_count, bcount_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bpos,   h_bucket_pos,   bpos_bytes,   cudaMemcpyHostToDevice));

    /* Copy constants */
    CUDA_CHECK(cudaMemcpyToSymbol(d_matrix, h_matrix, sizeof(h_matrix)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_query,  query,    (size_t)qlen));
    CUDA_CHECK(cudaMemcpyToSymbol(d_qlen,           &qlen,                sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_xdrop,          &opts->x_drop,        sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_min_score,      &opts->min_score,     sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_seed_score_min, &opts->seed_score_min,sizeof(int)));

    /* ----- Launch kernel ----- */
    dim3 grid(opts->num_blocks);
    dim3 block(opts->num_threads);
    diamond_kernel<<<grid, block>>>(d_db, d_lens, opts->num_sequences,
                                    opts->padded_length,
                                    d_bcount, d_bpos, d_results);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        free(h_bucket_count); free(h_bucket_pos);
        cudaFree(d_db); cudaFree(d_lens); cudaFree(d_bcount); cudaFree(d_bpos); cudaFree(d_results);
        return (int)err;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /* ----- Copy back ----- */
    CUDA_CHECK(cudaMemcpy(results, d_results, results_bytes, cudaMemcpyDeviceToHost));

    /* ----- Cleanup ----- */
    free(h_bucket_count);
    free(h_bucket_pos);
    cudaFree(d_db);
    cudaFree(d_lens);
    cudaFree(d_bcount);
    cudaFree(d_bpos);
    cudaFree(d_results);
    return 0;
}
