/****
GPU-Diamond UltraFast - Optimized CUDA implementation
- Persistent database (load once, query many)
- Early termination when MAX_HSPS reached
- Warp-level collaborative seed finding
- Score-based HSP priority queue
- Async streaming for overlap
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
    size_t         db_bytes;      /* Size of d_db */
    size_t         lens_bytes;    /* Size of d_lens */
    
    /* GPU work buffers (reused across queries) */
    DiamondResult* d_results;     /* Device results buffer */
    size_t         results_bytes; /* Size of d_results */
    
    /* CUDA streams for async execution */
    cudaStream_t   compute_stream;
    cudaStream_t   transfer_stream;
    
    /* Performance stats */
    int            queries_processed;
    float          total_kernel_time;
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
