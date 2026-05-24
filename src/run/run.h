/****
GPU-Diamond main execution pipeline
Like real Diamond's run/run.h
****/

#ifndef RUN_H
#define RUN_H

#include "../basic/gpu_diamond.h"
#include "../data/sequence.h"
#include "../stats/karlin_altschul.h"
#include "../output/output_format.h"

/* Search configuration (like Diamond's Search::Config) */
typedef struct {
    /* Scoring */
    int xdrop_ungapped;
    int xdrop_gapped;
    int min_score;
    int min_bitscore;
    double max_evalue;
    
    /* Sensitivity */
    int sensitivity;    /* 0=faster, 1=fast, 2=default, 3=sensitive */
    
    /* Filtering */
    int max_hsps;
    double min_id;
    double query_cover;
    double subject_cover;
    
    /* Masking */
    int mask_low_complexity;
    
    /* Output */
    OutputFormat output_format;
    int output_fields;
    
    /* Statistics */
    KarlinAltschul* ka;
    DBStats* db_stats;
    
    /* Execution */
    int num_blocks;
    int num_threads;
} SearchConfig;

/* Run a complete search (like Diamond's main pipeline) */
int run_search(const unsigned char* query, int qlen,
               Block* db,
               const SearchConfig* cfg,
               DiamondResult* results);

/* Setup and cleanup */
void run_init();
void run_cleanup();

#endif
