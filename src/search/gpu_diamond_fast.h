/****
GPU-Diamond UltraFast v2 - Production Optimized
Fixes all correctness and performance issues:
1. Correct seed_hits/extensions counters
2. Persistent query lookup buffers in GPUDiamondDB
3. Sparse seed index transfer (only non-empty buckets)
4. Real batch search with double buffering
5. Reduced local memory pressure (compressed diag table)
6. V100 auto-tuning
****/

#ifndef GPU_DIAMOND_FAST_H
#define GPU_DIAMOND_FAST_H

#include "../basic/gpu_diamond.h"
#include <cuda_runtime.h>

/* Persistent GPU database context - load once, reuse for many queries */
typedef struct {
    /* GPU-resident database (persistent across queries) */
    unsigned char* d_db;          /* Device database sequences */
    int*           d_lens;        /* Device sequence lengths */
    int            num_sequences; /* Total sequences in DB */
    int            padded_len;    /* Padded length per sequence */
    size_t         db_bytes;
    size_t         lens_bytes;
    
    /* GPU work buffers (reused across queries) */
    DiamondResult* d_results;
    size_t         results_bytes;
    
    /* Persistent query lookup buffers (reused across queries) */
    uint16_t*      d_sparse_hash;     /* Sparse bucket hashes */
    uint16_t*      d_sparse_count;    /* Hits per bucket */
    uint32_t*      d_sparse_offset;   /* Offset into positions */
    uint32_t*      d_sparse_positions; /* Flat positions array */
    size_t         max_sparse_buckets;
    size_t         max_positions;
    
    /* CUDA streams for async execution */
    cudaStream_t   compute_stream;
    cudaStream_t   transfer_stream;
    
    /* Auto-tuned parameters */
    int            tuned_num_blocks;
    int            tuned_num_threads;
    
    /* Performance stats */
    int            queries_processed;
    float          total_kernel_time;
    float          total_transfer_time;
} GPUDiamondDB;

/* Initialize persistent GPU database */
int gpudiamond_db_init(GPUDiamondDB* db, 
                       const unsigned char* db_padded,
                       const int* seq_lengths,
                       int num_sequences, 
                       int padded_len);

/* Destroy persistent GPU database */
void gpudiamond_db_destroy(GPUDiamondDB* db);

/* Search with persistent DB - ultrafast, no reallocation */
int gpudiamond_search_fast(GPUDiamondDB* db,
                           const unsigned char* query, 
                           int qlen,
                           const DiamondGPUOptions* opts,
                           DiamondResult* results);

/* Batch search multiple queries with async overlap */
int gpudiamond_search_batch(GPUDiamondDB* db,
                            const unsigned char** queries,
                            const int* qlens,
                            int num_queries,
                            const DiamondGPUOptions* opts,
                            DiamondResult** results);

/* Performance metrics */
void gpudiamond_print_stats(const GPUDiamondDB* db);

#endif
