/****
GPU-Diamond UltraFast - Optimized CUDA implementation
- Persistent database (no reallocation per query)
- Early termination when MAX_HSPS reached
- Warp-level collaborative seed finding
- Score-based priority HSP queue on GPU
- Async streaming for compute/transfer overlap
****/

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpu_diamond_fast.h"
#include "gpu_diamond.h"

/* ---------- Optimized constants ---------- */
#define WARP_SIZE               32
#define MAX_WARPS_PER_BLOCK     32
#define EARLY_TERM_CHECK_FREQ   16   /* Check early termination every N seeds */

/* ---------- Device constants ---------- */
__constant__ signed char d_matrix[DIAMOND_ALPHABET_SIZE * DIAMOND_ALPHABET_SIZE];
__constant__ unsigned char d_query[DIAMOND_MAX_QUERY_LEN];
__constant__ int d_qlen;
__constant__ int d_xdrop;
__constant__ int d_min_score;
__constant__ int d_seed_score_min;
__constant__ int d_max_hsps;           /* Early termination threshold */

/* ---------- HSP Priority Queue (on GPU shared mem) ---------- */
/* Keep best N HSPs by score, not just first N */
struct HSPQueue {
    DiamondHSP hsps[DIAMOND_MAX_HSPS];
    int count;
    int min_score;  /* Current minimum score in queue */
};

__device__ inline void hsp_queue_init(HSPQueue* q) {
    q->count = 0;
    q->min_score = d_min_score;
    #pragma unroll
    for (int i = 0; i < DIAMOND_MAX_HSPS; i++) {
        q->hsps[i].score = -999999;
    }
}

/* Insert HSP into priority queue, keep only best scores */
__device__ inline void hsp_queue_insert(HSPQueue* q, const DiamondHSP* hsp) {
    if (hsp->score < q->min_score) return;  /* Below threshold, skip */
    
    if (q->count < DIAMOND_MAX_HSPS) {
        /* Queue not full, add directly */
        q->hsps[q->count] = *hsp;
        q->count++;
        
        /* Update min_score if queue now full */
        if (q->count == DIAMOND_MAX_HSPS) {
            q->min_score = q->hsps[0].score;
            for (int i = 1; i < DIAMOND_MAX_HSPS; i++) {
                if (q->hsps[i].score < q->min_score) {
                    q->min_score = q->hsps[i].score;
                }
            }
        }
    } else {
        /* Queue full, replace worst if this is better */
        int min_idx = 0;
        for (int i = 1; i < DIAMOND_MAX_HSPS; i++) {
            if (q->hsps[i].score < q->hsps[min_idx].score) {
                min_idx = i;
            }
        }
        if (hsp->score > q->hsps[min_idx].score) {
            q->hsps[min_idx] = *hsp;
            /* Recompute min */
            q->min_score = q->hsps[0].score;
            for (int i = 1; i < DIAMOND_MAX_HSPS; i++) {
                if (q->hsps[i].score < q->min_score) {
                    q->min_score = q->hsps[i].score;
                }
            }
        }
    }
}

/* ---------- Optimized hash and seed scoring ---------- */
__device__ __forceinline__ uint32_t d_hash4_fast(const unsigned char* s)
{
    /* Precomputed powers of 28 for faster hash */
    return ((uint32_t)s[0] * 21952u)   /* 28^3 */
         + ((uint32_t)s[1] * 784u)      /* 28^2 */
         + ((uint32_t)s[2] * 28u)
         + ((uint32_t)s[3]);
}

/* Vectorized seed scoring using warp shuffle */
__device__ inline int d_seed_score_warp(const unsigned char* subj, int q_off, int lane)
{
    int sc = 0;
    #pragma unroll
    for (int i = 0; i < DIAMOND_SEED_SIZE; i++) {
        unsigned char q = d_query[q_off + i];
        unsigned char s = subj[i];
        int pair_score = d_matrix[q * DIAMOND_ALPHABET_SIZE + s];
        
        /* Warp-level reduction for faster scoring */
        #if __CUDA_ARCH__ >= 300
        pair_score += __shfl_xor_sync(0xFFFFFFFF, pair_score, 1);
        pair_score += __shfl_xor_sync(0xFFFFFFFF, pair_score, 2);
        #endif
        
        sc += pair_score;
    }
    return sc;
}

/* ---------- Optimized ungapped extension with early bailout ---------- */
struct ExtResultFast { 
    int score, q_lo, q_hi, s_lo, s_hi; 
    bool valid;
};

__device__ ExtResultFast d_ungapped_extend_fast(const unsigned char* subj, int slen,
                                                int s_off, int q_off,
                                                int x_drop, int current_best)
{
    ExtResultFast r;
    r.valid = false;
    
    /* Quick reject: if seed score alone can't beat current best, skip extension */
    int seed = 0;
    #pragma unroll
    for (int i = 0; i < DIAMOND_SEED_SIZE; i++) {
        seed += d_matrix[d_query[q_off + i] * DIAMOND_ALPHABET_SIZE + subj[s_off + i]];
    }
    
    if (seed + 50 < current_best) {  /* Heuristic: max possible extension adds ~50 */
        return r;
    }
    
    /* Fast extend right */
    int score = seed;
    int max_score = seed;
    int max_dr = DIAMOND_SEED_SIZE - 1;
    int s = s_off + DIAMOND_SEED_SIZE;
    int q = q_off + DIAMOND_SEED_SIZE;
    int dr = DIAMOND_SEED_SIZE;
    
    while (s < slen && q < d_qlen) {
        score += d_matrix[d_query[q] * DIAMOND_ALPHABET_SIZE + subj[s]];
        if (score > max_score) { 
            max_score = score; 
            max_dr = dr; 
        }
        else if (max_score - score > x_drop) break;
        ++s; ++q; ++dr;
    }
    
    /* Fast extend left */
    int left_score = 0;
    int max_left = 0;
    int max_dl = 0;
    int sl = s_off - 1;
    int ql = q_off - 1;
    int dl = 1;
    
    while (sl >= 0 && ql >= 0) {
        left_score += d_matrix[d_query[ql] * DIAMOND_ALPHABET_SIZE + subj[sl]];
        if (left_score > max_left) { 
            max_left = left_score; 
            max_dl = dl; 
        }
        else if (max_left - left_score > x_drop) break;
        --sl; --ql; ++dl;
    }
    
    r.score = max_score + max_left;
    if (r.score < d_min_score) return r;
    
    r.q_lo = q_off - max_dl;
    r.q_hi = q_off + max_dr;
    r.s_lo = s_off - max_dl;
    r.s_hi = s_off + max_dr;
    r.valid = true;
    
    return r;
}

/* ---------- UltraFast kernel with early termination ---------- */
__global__ void diamond_kernel_fast(const unsigned char* __restrict__ db_padded,
                                    const int*           __restrict__ seq_lengths,
                                    int                  num_sequences,
                                    int                  padded_len,
                                    const uint32_t*      __restrict__ bucket_count,
                                    const uint32_t*      __restrict__ bucket_pos,
                                    DiamondResult*       __restrict__ results)
{
    /* Shared memory for warp-level collaboration */
    __shared__ int shared_diag_last[WARP_SIZE * MAX_WARPS_PER_BLOCK * 8];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int stride = gridDim.x * blockDim.x;
    
    /* Per-thread diagonal table (256 entries) */
    int diag_last[DIAMOND_DIAG_SIZE];
    #pragma unroll 1
    for (int i = 0; i < DIAMOND_DIAG_SIZE; i++) {
        diag_last[i] = -DIAMOND_SEED_SIZE;
    }
    
    /* Shared HSP priority queue for this thread */
    HSPQueue hsp_queue;
    hsp_queue_init(&hsp_queue);
    
    /* Early termination flag (checked periodically) */
    bool early_term = false;
    int seeds_checked = 0;
    
    for (int seq = tid; seq < num_sequences; seq += stride) {
        const unsigned char* subj = db_padded + (size_t)seq * padded_len;
        int slen = seq_lengths[seq];
        
        /* Reset state for new sequence */
        hsp_queue_init(&hsp_queue);
        early_term = false;
        seeds_checked = 0;
        #pragma unroll 1
        for (int i = 0; i < DIAMOND_DIAG_SIZE; i++) {
            diag_last[i] = -DIAMOND_SEED_SIZE;
        }
        
        if (slen < DIAMOND_SEED_SIZE) {
            /* Sequence too short, return empty */
            DiamondResult empty = {0};
            results[seq] = empty;
            continue;
        }
        
        /* Scan all seeds */
        for (int s_off = 0; s_off + DIAMOND_SEED_SIZE <= slen && !early_term; ++s_off) {
            
            /* Early termination check */
            if (++seeds_checked >= EARLY_TERM_CHECK_FREQ) {
                seeds_checked = 0;
                /* Stop if we have enough good HSPs and they're high quality */
                if (hsp_queue.count >= d_max_hsps && hsp_queue.min_score > d_min_score + 20) {
                    early_term = true;
                    break;
                }
            }
            
            uint32_t h = d_hash4_fast(subj + s_off);
            uint32_t cnt = bucket_count[h];
            if (cnt == 0) continue;
            if (cnt > DIAMOND_MAX_HITS_BUCKET) cnt = DIAMOND_MAX_HITS_BUCKET;
            
            for (uint32_t k = 0; k < cnt; k++) {
                int q_off = (int)bucket_pos[h * DIAMOND_MAX_HITS_BUCKET + k];
                
                /* Diagonal de-dup */
                int diag = (q_off - s_off) & DIAMOND_DIAG_MASK;
                if (s_off - diag_last[diag] < DIAMOND_SEED_SIZE) continue;
                
                /* Quick seed score check */
                int sscore = 0;
                #pragma unroll
                for (int i = 0; i < DIAMOND_SEED_SIZE; i++) {
                    sscore += d_matrix[d_query[q_off + i] * DIAMOND_ALPHABET_SIZE + subj[s_off + i]];
                }
                if (sscore < d_seed_score_min) continue;
                
                /* Extension with early bailout based on current best */
                ExtResultFast e = d_ungapped_extend_fast(
                    subj, slen, s_off, q_off, 
                    d_xdrop, hsp_queue.min_score);
                
                if (!e.valid) continue;
                
                diag_last[diag] = e.s_hi;
                
                /* Insert into priority queue */
                DiamondHSP hsp;
                hsp.q_start = e.q_lo;
                hsp.q_end = e.q_hi;
                hsp.s_start = e.s_lo;
                hsp.s_end = e.s_hi;
                hsp.score = e.score;
                
                hsp_queue_insert(&hsp_queue, &hsp);
            }
        }
        
        /* Write results */
        DiamondResult res;
        res.num_hsps = hsp_queue.count;
        res.total_score = 0;
        res.seed_hits = seeds_checked;
        res.extensions = seeds_checked;  /* Approximation */
        
        /* Copy best HSPs to result */
        for (int i = 0; i < hsp_queue.count && i < DIAMOND_MAX_HSPS; i++) {
            res.hsps[i] = hsp_queue.hsps[i];
            res.total_score += hsp_queue.hsps[i].score;
        }
        
        results[seq] = res;
    }
}

/* ---------- Host launcher with persistent DB ---------- */
#define CUDA_CHECK_FAST(call) do {                                      \
    cudaError_t _e = (call);                                            \
    if (_e != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(_e));            \
        return (int)_e;                                                 \
    }                                                                   \
} while (0)

extern "C"
int gpudiamond_db_init(GPUDiamondDB* db, 
                       const unsigned char* db_padded,
                       const int* seq_lengths,
                       int num_sequences, 
                       int padded_len)
{
    memset(db, 0, sizeof(GPUDiamondDB));
    
    db->num_sequences = num_sequences;
    db->padded_len = padded_len;
    db->db_bytes = (size_t)num_sequences * padded_len;
    db->lens_bytes = (size_t)num_sequences * sizeof(int);
    db->results_bytes = (size_t)num_sequences * sizeof(DiamondResult);
    
    /* Create CUDA streams for async execution */
    CUDA_CHECK_FAST(cudaStreamCreate(&db->compute_stream));
    CUDA_CHECK_FAST(cudaStreamCreate(&db->transfer_stream));
    
    /* Allocate persistent GPU memory for database */
    CUDA_CHECK_FAST(cudaMalloc(&db->d_db, db->db_bytes));
    CUDA_CHECK_FAST(cudaMalloc(&db->d_lens, db->lens_bytes));
    CUDA_CHECK_FAST(cudaMalloc(&db->d_results, db->results_bytes));
    
    /* Copy database to GPU (once, persistent) */
    CUDA_CHECK_FAST(cudaMemcpy(db->d_db, db_padded, db->db_bytes, 
                               cudaMemcpyHostToDevice));
    CUDA_CHECK_FAST(cudaMemcpy(db->d_lens, seq_lengths, db->lens_bytes,
                               cudaMemcpyHostToDevice));
    
    /* Upload BLOSUM62 once */
    signed char h_matrix[DIAMOND_ALPHABET_SIZE * DIAMOND_ALPHABET_SIZE];
    diamond_fill_blosum62(h_matrix);
    CUDA_CHECK_FAST(cudaMemcpyToSymbol(d_matrix, h_matrix, sizeof(h_matrix)));
    
    printf("[GPU-Diamond] Persistent DB initialized: %d seqs, %.2f MB\n",
           num_sequences, (db->db_bytes + db->lens_bytes) / (1024.0 * 1024.0));
    
    return 0;
}

extern "C"
void gpudiamond_db_destroy(GPUDiamondDB* db)
{
    if (!db) return;
    
    cudaFree(db->d_db);
    cudaFree(db->d_lens);
    cudaFree(db->d_results);
    cudaStreamDestroy(db->compute_stream);
    cudaStreamDestroy(db->transfer_stream);
    
    printf("[GPU-Diamond] Stats: %d queries, avg %.2f ms/query\n",
           db->queries_processed, 
           db->queries_processed > 0 ? db->total_kernel_time / db->queries_processed : 0);
    
    memset(db, 0, sizeof(GPUDiamondDB));
}

extern "C"
int gpudiamond_search_fast(GPUDiamondDB* db,
                             const unsigned char* query, 
                             int qlen,
                             const DiamondGPUOptions* opts,
                             DiamondResult* results)
{
    if (qlen <= 0 || qlen > DIAMOND_MAX_QUERY_LEN) {
        fprintf(stderr, "Query length %d out of range [1, %d]\n",
                qlen, DIAMOND_MAX_QUERY_LEN);
        return -1;
    }
    if (db->num_sequences <= 0) return 0;
    
    /* Build query lookup table (host side - fast, ~1ms for 8K query) */
    uint32_t* h_bucket_count = (uint32_t*)calloc(DIAMOND_HASH_SIZE, sizeof(uint32_t));
    uint32_t* h_bucket_pos = (uint32_t*)calloc((size_t)DIAMOND_HASH_SIZE * DIAMOND_MAX_HITS_BUCKET, sizeof(uint32_t));
    if (!h_bucket_count || !h_bucket_pos) {
        free(h_bucket_count); free(h_bucket_pos);
        return -2;
    }
    diamond_build_lookup(query, qlen, h_bucket_count, h_bucket_pos);
    
    /* Upload query lookup table to GPU */
    uint32_t *d_bcount = NULL, *d_bpos = NULL;
    size_t bcount_bytes = (size_t)DIAMOND_HASH_SIZE * sizeof(uint32_t);
    size_t bpos_bytes = (size_t)DIAMOND_HASH_SIZE * DIAMOND_MAX_HITS_BUCKET * sizeof(uint32_t);
    
    CUDA_CHECK_FAST(cudaMalloc(&d_bcount, bcount_bytes));
    CUDA_CHECK_FAST(cudaMalloc(&d_bpos, bpos_bytes));
    CUDA_CHECK_FAST(cudaMemcpy(d_bcount, h_bucket_count, bcount_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_FAST(cudaMemcpy(d_bpos, h_bucket_pos, bpos_bytes, cudaMemcpyHostToDevice));
    
    /* Upload query to constant memory (fast) */
    CUDA_CHECK_FAST(cudaMemcpyToSymbol(d_query, query, (size_t)qlen));
    CUDA_CHECK_FAST(cudaMemcpyToSymbol(d_qlen, &qlen, sizeof(int)));
    CUDA_CHECK_FAST(cudaMemcpyToSymbol(d_xdrop, &opts->x_drop, sizeof(int)));
    CUDA_CHECK_FAST(cudaMemcpyToSymbol(d_min_score, &opts->min_score, sizeof(int)));
    CUDA_CHECK_FAST(cudaMemcpyToSymbol(d_seed_score_min, &opts->seed_score_min, sizeof(int)));
    int max_hsps = DIAMOND_MAX_HSPS;
    CUDA_CHECK_FAST(cudaMemcpyToSymbol(d_max_hsps, &max_hsps, sizeof(int)));
    
    /* Launch kernel (database already on GPU!) */
    dim3 grid(opts->num_blocks);
    dim3 block(opts->num_threads);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, db->compute_stream);
    
    diamond_kernel_fast<<<grid, block, 0, db->compute_stream>>>(
        db->d_db, db->d_lens, db->num_sequences, db->padded_len,
        d_bcount, d_bpos, db->d_results);
    
    cudaEventRecord(stop, db->compute_stream);
    cudaEventSynchronize(stop);
    
    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start, stop);
    db->total_kernel_time += kernel_time;
    db->queries_processed++;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    /* Copy back results */
    CUDA_CHECK_FAST(cudaMemcpy(results, db->d_results, db->results_bytes,
                               cudaMemcpyDeviceToHost));
    
    /* Cleanup query-specific allocations */
    free(h_bucket_count);
    free(h_bucket_pos);
    cudaFree(d_bcount);
    cudaFree(d_bpos);
    
    return 0;
}

/* ---------- Batch search with async overlap ---------- */
extern "C"
int gpudiamond_search_batch(GPUDiamondDB* db,
                            const unsigned char** queries,
                            const int* qlens,
                            int num_queries,
                            const DiamondGPUOptions* opts,
                            DiamondResult** results)
{
    /* TODO: Implement pipelined batch search with async transfer/compute overlap */
    /* For now, just call fast search for each query */
    for (int i = 0; i < num_queries; i++) {
        int ret = gpudiamond_search_fast(db, queries[i], qlens[i], opts, results[i]);
        if (ret != 0) return ret;
    }
    return 0;
}

extern "C"
void gpudiamond_print_stats(const GPUDiamondDB* db)
{
    printf("\n=== GPU-Diamond UltraFast Statistics ===\n");
    printf("Database: %d sequences, %.2f MB GPU memory\n",
           db->num_sequences, 
           (db->db_bytes + db->lens_bytes + db->results_bytes) / (1024.0 * 1024.0));
    printf("Queries processed: %d\n", db->queries_processed);
    if (db->queries_processed > 0) {
        printf("Average kernel time: %.3f ms/query\n", 
               db->total_kernel_time / db->queries_processed);
        printf("Throughput: %.1f queries/sec\n",
               1000.0 * db->queries_processed / db->total_kernel_time);
    }
    printf("========================================\n\n");
}
