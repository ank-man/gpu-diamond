#ifndef GPU_DIAMOND_H
#define GPU_DIAMOND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------- Configuration ---------------- */
#define DIAMOND_ALPHABET_SIZE   28           /* matches BLOSUM62 padded layout */
#define DIAMOND_SEED_SIZE       4            /* contiguous 4-mer seed         */
/* Hash space = 28^4 = 614656 */
#define DIAMOND_HASH_SIZE       614656
#define DIAMOND_MAX_HITS_BUCKET 32           /* max query positions per seed */
#define DIAMOND_MAX_HSPS        32           /* max HSPs reported per subject*/
#define DIAMOND_DIAG_SIZE       256          /* per-thread diagonal table (must be power of 2) */
#define DIAMOND_DIAG_MASK       (DIAMOND_DIAG_SIZE - 1)
#define DIAMOND_MAX_QUERY_LEN   8192

/* ---------------- Data structures ---------------- */
typedef struct {
    int q_start;
    int q_end;
    int s_start;
    int s_end;
    int score;
} DiamondHSP;

typedef struct {
    DiamondHSP hsps[DIAMOND_MAX_HSPS];
    int        num_hsps;
    int        total_score;
    int        seed_hits;
    int        extensions;
} DiamondResult;

typedef struct {
    int    num_blocks;       /* CUDA grid X dimension                 */
    int    num_threads;      /* CUDA block X dimension                */
    int    num_sequences;    /* number of subject sequences           */
    int    padded_length;    /* per-sequence padded length (>= max)   */
    int    x_drop;           /* X-dropoff for ungapped extension      */
    int    min_score;        /* minimum HSP score to report           */
    int    seed_score_min;   /* minimum seed match score              */
} DiamondGPUOptions;

/* ---------------- Public API (CPU side) ---------------- */
/* Encode an ASCII protein sequence into NCBIstdaa-like codes (0..27).
 * Returns a malloc'd byte array of length len. */
unsigned char* diamond_encode_protein(const char* ascii, int len);

/* Build BLOSUM62 substitution matrix in row-major
 * matrix[q*28 + s]. Buffer must hold 28*28 = 784 chars. */
void diamond_fill_blosum62(signed char* matrix);

/* Hash a 4-mer of encoded amino acids (each 0..27).
 * Returns hash in [0, DIAMOND_HASH_SIZE). */
uint32_t diamond_hash4(const unsigned char* seq);

/* Build the query lookup table:
 *   bucket_count[h]  = number of query positions with hash h (clamped to MAX)
 *   bucket_pos[h*MAX + i] = i-th query offset for hash h
 * The arrays must be allocated by the caller with sizes:
 *   bucket_count: DIAMOND_HASH_SIZE   (uint32_t)
 *   bucket_pos:   DIAMOND_HASH_SIZE * DIAMOND_MAX_HITS_BUCKET (uint32_t)
 */
void diamond_build_lookup(const unsigned char* query, int qlen,
                          uint32_t* bucket_count, uint32_t* bucket_pos);

/* Run the GPU Diamond search. results[] must hold num_sequences entries.
 * Returns 0 on success, non-zero on CUDA error. */
int diamond_gpu_search(const unsigned char* query, int qlen,
                       const unsigned char* db_padded,
                       const int*           seq_lengths,
                       const DiamondGPUOptions* opts,
                       DiamondResult*       results);

#ifdef __cplusplus
}
#endif

#endif /* GPU_DIAMOND_H */
