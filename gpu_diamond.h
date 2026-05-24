#ifndef GPU_DIAMOND_H
#define GPU_DIAMOND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * GPU-Diamond v2.0 - Enhanced with real Diamond features
 * Based on github.com/bbuchfink/diamond
 * ============================================================ */

/* ---------------- Configuration ---------------- */
#define DIAMOND_ALPHABET_SIZE       28
#define DIAMOND_MAX_QUERY_LEN       8192
#define DIAMOND_MAX_SEED_SHAPES     16          /* Multiple spaced seeds */
#define DIAMOND_MAX_SEED_WEIGHT     10          /* Max '1's in seed pattern */
#define DIAMOND_MAX_SEED_LENGTH     32          /* Max pattern length */
#define DIAMOND_MAX_HITS_BUCKET     32
#define DIAMOND_MAX_HSPS            32
#define DIAMOND_DIAG_SIZE           256
#define DIAMOND_DIAG_MASK           (DIAMOND_DIAG_SIZE - 1)
#define DIAMOND_SEED_SIZE           4           /* Default contiguous seed size */
#define DIAMOND_HASH_SIZE           614656      /* 28^4 for contiguous seeds */

/* Sensitivity modes matching real Diamond */
typedef enum {
    DIAMOND_SENSITIVITY_FASTER = 0,     /* --faster */
    DIAMOND_SENSITIVITY_FAST,           /* --fast */
    DIAMOND_SENSITIVITY_DEFAULT,        /* default */
    DIAMOND_SENSITIVITY_SENSITIVE,      /* --sensitive */
    DIAMOND_SENSITIVITY_MORE_SENSITIVE, /* --more-sensitive */
    DIAMOND_SENSITIVITY_VERY_SENSITIVE, /* --very-sensitive */
    DIAMOND_SENSITIVITY_ULTRA_SENSITIVE /* --ultra-sensitive */
} DiamondSensitivity;

/* Gapped extension modes */
typedef enum {
    DIAMOND_GAP_NONE = 0,       /* Ungapped only */
    DIAMOND_GAP_BANDED_FAST,    /* Fast banded SW */
    DIAMOND_GAP_BANDED_SLOW     /* More sensitive banded SW */
} DiamondGapMode;

/* Output formats */
typedef enum {
    DIAMOND_OUTPUT_TEXT = 0,    /* Human readable */
    DIAMOND_OUTPUT_TABULAR,     /* BLAST tabular */
    DIAMOND_OUTPUT_PAIRWISE,    /* BLAST pairwise */
    DIAMOND_OUTPUT_SAM,         /* SAM format */
    DIAMOND_OUTPUT_PAF          /* PAF format (minimap2 style) */
} DiamondOutputFormat;

/* ---------------- Spaced Seed Structure ---------------- */
typedef struct {
    char     pattern[DIAMOND_MAX_SEED_LENGTH + 1];  /* e.g. "111101110111" */
    int      weight;                                 /* Number of '1's */
    int      length;                                 /* Pattern length */
    int      positions[DIAMOND_MAX_SEED_WEIGHT];     /* Positions of '1's */
    uint32_t mask;                                   /* Bit mask */
} DiamondSeedShape;

/* ---------------- HSP with gapped alignment info ---------------- */
typedef struct {
    /* Ungapped coordinates */
    int q_start;
    int q_end;
    int s_start;
    int s_end;
    int ungapped_score;
    
    /* Gapped alignment info */
    int gapped_q_start;
    int gapped_q_end;
    int gapped_s_start;
    int gapped_s_end;
    int gapped_score;
    
    /* Statistics */
    double evalue;
    double bitscore;
    int    length;          /* Alignment length */
    int    nident;          /* Number of identical matches */
    int    ngaps;           /* Number of gaps */
    float  pident;          /* Percent identity */
    float  qcov;            /* Query coverage */
    float  scov;            /* Subject coverage */
} DiamondHSP;

typedef struct {
    DiamondHSP  hsps[DIAMOND_MAX_HSPS];
    int         num_hsps;
    int         total_ungapped_score;
    int         total_gapped_score;
    int         seed_hits;
    int         extensions;
    int         gapped_extensions;
    char        sseq_name[256];     /* Subject sequence name */
    int         sseq_len;
} DiamondResult;

/* ---------------- Statistics ---------------- */
typedef struct {
    double lambda;          /* Karlin-Altschul lambda */
    double K;               /* Karlin-Altschul K */
    double H;               /* Relative entropy */
    int    gap_open;        /* Gap open penalty */
    int    gap_extend;      /* Gap extend penalty */
} DiamondKarlinAltschul;

typedef struct {
    int64_t db_size;        /* Total database residues */
    int     db_seqs;        /* Number of sequences in db */
    double  search_space;   /* Effective search space */
} DiamondDBStats;

/* ---------------- GPU Options ---------------- */
typedef struct {
    /* Execution */
    int     num_blocks;
    int     num_threads;
    int     num_sequences;
    int     padded_length;
    
    /* Sensitivity */
    DiamondSensitivity sensitivity;
    DiamondGapMode     gap_mode;
    
    /* Seeds */
    DiamondSeedShape* seed_shapes;  /* Array of spaced seed patterns */
    int               num_shapes;   /* Number of seed shapes (1-16) */
    int               shape_idx;    /* Current shape index for lookup */
    
    /* Scoring */
    int     x_drop_ungapped;    /* X-dropoff for ungapped extension */
    int     x_drop_gapped;      /* X-dropoff for gapped extension */
    int     min_score;          /* Minimum score to report */
    int     min_bitscore;       /* Minimum bit score */
    double  max_evalue;         /* Maximum E-value threshold */
    int     gap_open;           /* Gap open penalty (BLOSUM62 default: 11) */
    int     gap_extend;         /* Gap extend penalty (BLOSUM62 default: 1) */
    int     band_width;         /* Band width for banded SW */
    
    /* Filtering */
    int     max_hsps_per_target;    /* Keep top N HSPs per target (0=all) */
    float   min_qcov;               /* Minimum query coverage */
    float   min_scov;               /* Minimum subject coverage */
    float   min_pident;             /* Minimum percent identity */
    
    /* Masking */
    int     mask_low_complexity;    /* Enable SEG masking */
    int     mask_window;          /* Masking window size */
    
    /* Output */
    DiamondOutputFormat output_format;
    int                 output_fields;  /* Bitmask of fields to output */
    
    /* Stats */
    DiamondDBStats*     db_stats;
    DiamondKarlinAltschul* ka_params;
} DiamondGPUOptions;

/* ---------------- Public API ---------------- */

/* Encode protein sequence */
unsigned char* diamond_encode_protein(const char* ascii, int len);

/* Build BLOSUM62 matrix */
void diamond_fill_blosum62(signed char* matrix);

/* Create a spaced seed shape from pattern string */
int diamond_seed_shape_init(DiamondSeedShape* shape, const char* pattern);

/* Hash a seed using a spaced seed shape */
uint64_t diamond_hash_seed_spaced(const unsigned char* seq, 
                                   const DiamondSeedShape* shape);

/* Hash for contiguous k-mer */
uint32_t diamond_hash4(const unsigned char* seq);

/* Build lookup table for a specific seed shape */
void diamond_build_lookup_spaced(const unsigned char* query, int qlen,
                                  const DiamondSeedShape* shape,
                                  uint32_t* bucket_count, 
                                  uint32_t* bucket_pos);

/* Legacy lookup build (contiguous 4-mer) */
void diamond_build_lookup(const unsigned char* query, int qlen,
                          uint32_t* bucket_count, uint32_t* bucket_pos);

/* SEG low-complexity masking */
void diamond_mask_seg(unsigned char* seq, int len, int window, int trigger);

/* Calculate Karlin-Altschul parameters for BLOSUM62 */
void diamond_init_karlin_altschul(DiamondKarlinAltschul* ka, 
                                   int gap_open, int gap_extend);

/* Compute E-value from raw score */
double diamond_evalue(int score, int query_len, 
                       const DiamondDBStats* db_stats,
                       const DiamondKarlinAltschul* ka);

/* Compute bit score from raw score */
double diamond_bitscore(int score, const DiamondKarlinAltschul* ka);

/* Run GPU search */
int diamond_gpu_search(const unsigned char* query, int qlen,
                       const unsigned char* db_padded,
                       const int*           seq_lengths,
                       const char**         seq_names,
                       const DiamondGPUOptions* opts,
                       DiamondResult*       results);

/* Compute statistics on results (E-values, coverage, etc.) */
void diamond_compute_stats(DiamondResult* results, int num_results,
                           int query_len,
                           const DiamondGPUOptions* opts);

/* Sort and cull HSPs per target */
void diamond_cull_hsps(DiamondResult* result, int max_hsps);

/* Output functions */
void diamond_output_header(DiamondOutputFormat fmt, int fields);
void diamond_output_result(const DiamondResult* result, int result_idx,
                          const char* qname, int qlen,
                          DiamondOutputFormat fmt, int fields);
void diamond_output_footer(DiamondOutputFormat fmt);

/* Sensitivity presets */
void diamond_set_sensitivity(DiamondGPUOptions* opts, DiamondSensitivity s);

/* ---------------- Utility ---------------- */
/* Get default shape patterns for sensitivity level */
const char** diamond_get_default_shapes(DiamondSensitivity s, int* count);

/* Compute percent identity from alignment */
float diamond_calc_pident(const unsigned char* query, int qstart,
                          const unsigned char* subject, int sstart,
                          int aln_len, const signed char* blosum62);

/* Compute coverage */
float diamond_calc_coverage(int aln_start, int aln_end, int seq_len);

#ifdef __cplusplus
}
#endif

#endif /* GPU_DIAMOND_H */
