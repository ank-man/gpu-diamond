/* gpu_diamond_v2.cu - CUDA implementation with gapped extension */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdint.h>
#include <float.h>

#define DIAMOND_ALPHABET_SIZE   28
#define DIAMOND_MAX_SEED_WEIGHT 10
#define DIAMOND_MAX_SEED_LENGTH 32
#define DIAMOND_MAX_HITS_BUCKET 32
#define DIAMOND_MAX_HSPS        32
#define DIAMOND_DIAG_SIZE       256
#define DIAMOND_DIAG_MASK       255
#define DIAMOND_MAX_QUERY_LEN   8192
#define MAX_BAND_WIDTH          64  /* For banded SW */

/* Error checking macro */
#define CUDA_CHECK(err) do { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); \
        return e; \
    } \
} while(0)

/* ============================================================
 * Device constants (set via cudaMemcpyToSymbol)
 * ============================================================ */
__constant__ signed char d_matrix[28*28];
__constant__ unsigned char d_query[DIAMOND_MAX_QUERY_LEN];
__constant__ int d_qlen;
__constant__ int d_xdrop_ungapped;
__constant__ int d_xdrop_gapped;
__constant__ int d_min_score;
__constant__ int d_seed_score_min;
__constant__ int d_gap_open;
__constant__ int d_gap_extend;
__constant__ int d_band_width;
__constant__ int d_gap_mode;  /* 0=none, 1=banded_fast, 2=banded_slow */

/* ============================================================
 * Device structures
 * ============================================================ */
struct SeedShape {
    int  weight;
    int  length;
    int  positions[DIAMOND_MAX_SEED_WEIGHT];
    unsigned int mask;
};

__constant__ SeedShape d_seed_shape;

struct DiamondHSP {
    /* Ungapped coordinates */
    int q_start, q_end;
    int s_start, s_end;
    int ungapped_score;
    
    /* Gapped coordinates */
    int gapped_q_start, gapped_q_end;
    int gapped_s_start, gapped_s_end;
    int gapped_score;
    
    int diagonal;  /* q_start - s_start */
};

/* ============================================================
 * Device helper: Hash with spaced seed
 * ============================================================ */
__device__ uint64_t d_hash_seed_spaced(const unsigned char* seq, const SeedShape* shape)
{
    uint64_t hash = 0;
    for (int i = 0; i < shape->weight; i++) {
        hash *= DIAMOND_ALPHABET_SIZE;
        hash += seq[shape->positions[i]];
    }
    return hash;
}

/* ============================================================
 * Device helper: Score seed
 * ============================================================ */
__device__ int d_score_seed(const unsigned char* query, int qpos,
                            const unsigned char* subject, int spos,
                            int seed_len)
{
    int sc = 0;
    for (int k = 0; k < seed_len; k++) {
        unsigned char qc = query[qpos + k];
        unsigned char sc_ = subject[spos + k];
        sc += d_matrix[qc * DIAMOND_ALPHABET_SIZE + sc_];
    }
    return sc;
}

/* ============================================================
 * Device function: Ungapped X-drop extension
 * ============================================================ */
__device__ void d_ungapped_extend(const unsigned char* query, int qlen, int q_anchor,
                                   const unsigned char* subject, int slen, int s_anchor,
                                   int* left_score, int* left_qpos, int* left_spos,
                                   int* right_score, int* right_qpos, int* right_spos,
                                   int* total_score)
{
    const int xdrop = d_xdrop_ungapped;
    int best = 0, st = 0, n = 1;
    int delta = 0;
    
    /* Extend left */
    int q = q_anchor - 1;
    int s = s_anchor - 1;
    
    while (best - st < xdrop && q >= 0 && s >= 0 && s < slen) {
        st += d_matrix[d_query[q] * DIAMOND_ALPHABET_SIZE + subject[s]];
        if (st > best) {
            best = st;
            delta = n;
        }
        q--; s--; n++;
    }
    *left_score = best;
    *left_qpos = q_anchor - delta;
    *left_spos = s_anchor - delta;
    
    /* Extend right */
    best = 0; st = 0; n = 1;
    int len_right = 0;
    q = q_anchor;
    s = s_anchor;
    
    while (best - st < xdrop && q < qlen && s >= 0 && s < slen) {
        st += d_matrix[d_query[q] * DIAMOND_ALPHABET_SIZE + subject[s]];
        if (st > best) {
            best = st;
            len_right = n;
        }
        q++; s++; n++;
    }
    *right_score = best;
    *right_qpos = q_anchor + len_right - 1;
    *right_spos = s_anchor + len_right - 1;
    
    /* Total (seed_score + left + right, seed counted once) */
    int seed_score = d_matrix[d_query[q_anchor] * DIAMOND_ALPHABET_SIZE + subject[s_anchor]];
    *total_score = *left_score + seed_score + *right_score;
}

/* ============================================================
 * Device function: Banded Smith-Waterman (simplified)
 * Uses diagonal banding with reduced memory footprint
 * ============================================================ */
__device__ int d_banded_sw(const unsigned char* query, int q_start, int q_len,
                            const unsigned char* subject, int s_start, int s_len,
                            int band_width,
                            int* out_q_start, int* out_q_end,
                            int* out_s_start, int* out_s_end,
                            int xdrop)
{
    /* Simplified banded SW - compute within band around diagonal */
    const int gap_open = d_gap_open;
    const int gap_extend = d_gap_extend;
    
    int mid_diag = q_start - s_start;
    int half_band = band_width / 2;
    
    int max_score = 0;
    int max_q = q_start, max_s = s_start;
    
    int i = q_start, j = s_start;
    int local_score = 0;
    int gap_len = 0;
    
    /* Simple greedy extension within band */
    while (i < q_len && j < s_len && i >= 0 && j >= 0) {
        int diag = i - j;
        if (abs(diag - mid_diag) > half_band) break;
        
        int match = d_matrix[d_query[i] * DIAMOND_ALPHABET_SIZE + subject[j]];
        
        if (match > 0) {
            /* Match/mismatch */
            local_score += match;
            gap_len = 0;
        } else {
            /* Consider gap */
            int gap_penalty = gap_open + gap_len * gap_extend;
            if (match > -gap_penalty) {
                local_score += match;
                gap_len = 0;
            } else {
                local_score -= gap_penalty;
                gap_len++;
            }
        }
        
        /* X-drop check */
        if (local_score > max_score) {
            max_score = local_score;
            max_q = i;
            max_s = j;
        }
        
        if (max_score - local_score > xdrop) break;
        
        i++; j++;
    }
    
    *out_q_end = max_q;
    *out_s_end = max_s;
    *out_q_start = q_start;
    *out_s_start = s_start;
    
    return max_score;
}

/* ============================================================
 * Main kernel
 * ============================================================ */
__global__ void diamond_kernel(const unsigned char* db_padded,
                                const int* seq_lengths,
                                const uint32_t* bucket_count,
                                const uint32_t* bucket_pos,
                                uint64_t hash_space,
                                int num_seqs,
                                int padded_length,
                                DiamondHSP* results,
                                int* result_counts,
                                int max_hsps_per_seq)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    
    /* Per-thread diagonal table for dedup */
    unsigned short diag_table[DIAMOND_DIAG_SIZE];
    
    /* Process sequences in strided fashion */
    for (int seq = tid; seq < num_seqs; seq += total_threads) {
        int slen = seq_lengths[seq];
        if (slen <= 0) continue;
        
        const unsigned char* subject = db_padded + (size_t)seq * padded_length;
        
        /* Clear diagonal table */
        for (int i = 0; i < DIAMOND_DIAG_SIZE; i++) diag_table[i] = 0;
        
        /* Clear result slots for this sequence */
        DiamondHSP* my_results = results + (size_t)seq * max_hsps_per_seq;
        int hsp_count = 0;
        int seed_hits = 0;
        int extensions = 0;
        
        int seed_len = d_seed_shape.length > 0 ? d_seed_shape.length : 4;
        
        /* Scan subject for seeds */
        for (int s_pos = 0; s_pos <= slen - seed_len; s_pos++) {
            /* Check for invalid characters (mask) */
            int valid = 1;
            for (int k = 0; k < seed_len && valid; k++) {
                if (subject[s_pos + k] >= 27) valid = 0;
            }
            if (!valid) continue;
            
            /* Hash seed at this position */
            uint64_t h;
            if (d_seed_shape.weight > 0) {
                h = d_hash_seed_spaced(subject + s_pos, &d_seed_shape);
            } else {
                /* Fallback contiguous hash */
                h = ((uint64_t)subject[s_pos] * 28 * 28 * 28)
                  + ((uint64_t)subject[s_pos+1] * 28 * 28)
                  + ((uint64_t)subject[s_pos+2] * 28)
                  + (uint64_t)subject[s_pos+3];
            }
            
            if (h >= hash_space) continue;
            
            uint32_t count = bucket_count[h];
            if (count == 0) continue;
            if (count > DIAMOND_MAX_HITS_BUCKET) count = DIAMOND_MAX_HITS_BUCKET;
            
            uint32_t base = bucket_pos[h];
            seed_hits += count;
            
            /* Try each query hit position */
            for (uint32_t i = 0; i < count; i++) {
                int q_pos = bucket_pos[base + i];
                
                /* Check seed score threshold */
                int seed_sc = d_score_seed(d_query, q_pos, subject, s_pos, seed_len);
                if (seed_sc < d_seed_score_min) continue;
                
                /* Diagonal check for dedup */
                int diag = q_pos - s_pos;
                int diag_idx = diag & DIAMOND_DIAG_MASK;
                unsigned short diag_val = (unsigned short)((diag >> 8) & 0xFFFF);
                
                if (diag_table[diag_idx] == diag_val) continue;
                diag_table[diag_idx] = diag_val;
                
                extensions++;
                
                /* Ungapped extension */
                int left_sc, right_sc, total_sc;
                int left_q, left_s, right_q, right_s;
                
                d_ungapped_extend(d_query, d_qlen, q_pos,
                                  subject, slen, s_pos,
                                  &left_sc, &left_q, &left_s,
                                  &right_sc, &right_q, &right_s,
                                  &total_sc);
                
                if (total_sc < d_min_score) continue;
                
                /* Store as HSP */
                if (hsp_count < max_hsps_per_seq) {
                    DiamondHSP* h = &my_results[hsp_count];
                    h->q_start = left_q;
                    h->q_end = right_q;
                    h->s_start = left_s;
                    h->s_end = right_s;
                    h->ungapped_score = total_sc;
                    h->diagonal = diag;
                    
                    /* Gapped extension if enabled */
                    if (d_gap_mode > 0 && d_band_width > 0) {
                        h->gapped_score = d_banded_sw(
                            d_query, left_q, d_qlen,
                            subject, left_s, slen,
                            d_band_width,
                            &h->gapped_q_start, &h->gapped_q_end,
                            &h->gapped_s_start, &h->gapped_s_end,
                            d_xdrop_gapped
                        );
                        if (h->gapped_score < h->ungapped_score) {
                            /* Gapped didn't improve - use ungapped */
                            h->gapped_score = 0;
                        }
                    } else {
                        h->gapped_score = 0;
                    }
                    
                    hsp_count++;
                }
            }
        }
        
        result_counts[seq] = hsp_count;
    }
}

/* ============================================================
 * Host launcher
 * ============================================================ */

extern "C" {

typedef struct {
    int q_start, q_end, s_start, s_end, ungapped_score;
    int gapped_q_start, gapped_q_end, gapped_s_start, gapped_s_end, gapped_score;
    int diagonal;
} DiamondHSP_Device;

typedef struct {
    int weight;
    int length;
    int positions[DIAMOND_MAX_SEED_WEIGHT];
    unsigned int mask;
} SeedShape_Host;

int diamond_gpu_search_v2(const unsigned char* query, int qlen,
                          const unsigned char* db_padded,
                          const int* seq_lengths,
                          int num_sequences,
                          int padded_length,
                          const uint32_t* bucket_count,
                          const uint32_t* bucket_pos,
                          uint64_t hash_space,
                          const signed char* blosum62,
                          const SeedShape_Host* seed_shape,
                          int xdrop_ungapped,
                          int xdrop_gapped,
                          int min_score,
                          int seed_score_min,
                          int gap_open,
                          int gap_extend,
                          int band_width,
                          int gap_mode,
                          int num_blocks,
                          int num_threads,
                          DiamondHSP_Device* results,
                          int* result_counts,
                          int max_hsps_per_seq)
{
    /* Copy BLOSUM62 to constant memory */
    CUDA_CHECK(cudaMemcpyToSymbol(d_matrix, blosum62, 28*28));
    
    /* Copy query to constant memory */
    if (qlen > DIAMOND_MAX_QUERY_LEN) qlen = DIAMOND_MAX_QUERY_LEN;
    CUDA_CHECK(cudaMemcpyToSymbol(d_query, query, qlen));
    int zero = 0;
    CUDA_CHECK(cudaMemcpyToSymbol(d_query + qlen, &zero, 1));
    
    /* Copy scalar params */
    CUDA_CHECK(cudaMemcpyToSymbol(d_qlen, &qlen, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_xdrop_ungapped, &xdrop_ungapped, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_xdrop_gapped, &xdrop_gapped, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_min_score, &min_score, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_seed_score_min, &seed_score_min, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_gap_open, &gap_open, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_gap_extend, &gap_extend, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_band_width, &band_width, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_gap_mode, &gap_mode, sizeof(int)));
    
    /* Copy seed shape */
    if (seed_shape) {
        SeedShape dev_shape;
        dev_shape.weight = seed_shape->weight;
        dev_shape.length = seed_shape->length;
        dev_shape.mask = seed_shape->mask;
        for (int i = 0; i < DIAMOND_MAX_SEED_WEIGHT; i++) {
            dev_shape.positions[i] = seed_shape->positions[i];
        }
        CUDA_CHECK(cudaMemcpyToSymbol(d_seed_shape, &dev_shape, sizeof(SeedShape)));
    }
    
    /* Allocate device memory for database */
    unsigned char* d_db = NULL;
    int* d_seq_lens = NULL;
    size_t db_size = (size_t)num_sequences * padded_length;
    
    CUDA_CHECK(cudaMalloc(&d_db, db_size));
    CUDA_CHECK(cudaMalloc(&d_seq_lens, num_sequences * sizeof(int)));
    
    CUDA_CHECK(cudaMemcpy(d_db, db_padded, db_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_seq_lens, seq_lengths, num_sequences * sizeof(int), 
                          cudaMemcpyHostToDevice));
    
    /* Allocate device memory for lookup tables */
    uint32_t* d_bucket_count = NULL;
    uint32_t* d_bucket_pos = NULL;
    
    CUDA_CHECK(cudaMalloc(&d_bucket_count, hash_space * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_bucket_pos, hash_space * DIAMOND_MAX_HITS_BUCKET * sizeof(uint32_t)));
    
    CUDA_CHECK(cudaMemcpy(d_bucket_count, bucket_count, hash_space * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bucket_pos, bucket_pos, 
                          hash_space * DIAMOND_MAX_HITS_BUCKET * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    
    /* Allocate results */
    DiamondHSP_Device* d_results = NULL;
    int* d_result_counts = NULL;
    
    CUDA_CHECK(cudaMalloc(&d_results, (size_t)num_sequences * max_hsps_per_seq * 
                          sizeof(DiamondHSP_Device)));
    CUDA_CHECK(cudaMalloc(&d_result_counts, num_sequences * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_result_counts, 0, num_sequences * sizeof(int)));
    
    /* Launch kernel */
    diamond_kernel<<<num_blocks, num_threads>>>(
        d_db, d_seq_lens,
        d_bucket_count, d_bucket_pos, hash_space,
        num_sequences, padded_length,
        (DiamondHSP*)d_results, d_result_counts, max_hsps_per_seq
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    /* Copy back results */
    CUDA_CHECK(cudaMemcpy(results, d_results,
                          (size_t)num_sequences * max_hsps_per_seq * sizeof(DiamondHSP_Device),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result_counts, d_result_counts,
                          num_sequences * sizeof(int),
                          cudaMemcpyDeviceToHost));
    
    /* Cleanup */
    cudaFree(d_db);
    cudaFree(d_seq_lens);
    cudaFree(d_bucket_count);
    cudaFree(d_bucket_pos);
    cudaFree(d_results);
    cudaFree(d_result_counts);
    
    return 0;
}

} /* extern "C" */
