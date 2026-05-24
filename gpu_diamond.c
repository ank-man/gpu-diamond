/* gpu_diamond_v2.c - CPU-side implementation with Diamond features */
#include "gpu_diamond_v2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* Standard amino acid encoding: A-Z -> 0-25, plus special codes */
static const unsigned char AA_MAP[256] = {
    ['A']=0,  ['B']=1,  ['C']=2,  ['D']=3,  ['E']=4,  ['F']=5,  ['G']=6,  ['H']=7,
    ['I']=8,  ['J']=9,  ['K']=10, ['L']=11, ['M']=12, ['N']=13, ['O']=14, ['P']=15,
    ['Q']=16, ['R']=17, ['S']=18, ['T']=19, ['U']=20, ['V']=21, ['W']=22, ['X']=23,
    ['Y']=24, ['Z']=25, ['a']=0,  ['b']=1,  ['c']=2,  ['d']=3,  ['e']=4,  ['f']=5,
    ['g']=6,  ['h']=7,  ['i']=8,  ['j']=9,  ['k']=10, ['l']=11, ['m']=12, ['n']=13,
    ['o']=14, ['p']=15, ['q']=16, ['r']=17, ['s']=18, ['t']=19, ['u']=20, ['v']=21,
    ['w']=22, ['x']=23, ['y']=24, ['z']=25, ['*']=26, ['-']=27
};

/* BLOSUM62 matrix (28x28, row-major) */
static const signed char BLOSUM62[28*28] = {
     4,-2, 0,-2,-1,-2, 0,-2,-1,-1,-1,-1,-1,-2, 0,-1,-1,-1, 1, 0,-4,-1,-2,-1,-2,-1, 0, 0,
    -2, 4,-3, 4, 1,-3,-1, 0,-3, 0, 0,-4, 0, 0, 3,-1,-1, 0, 0,-1,-4,-3,-3, 0,-3, 1, 0, 0,
     0,-3, 9,-3,-4,-2,-3,-3,-1,-3,-3,-1,-1,-3,-3,-3,-3,-3,-1,-1,-4,-1,-2,-1,-2,-3, 0, 0,
    -2, 4,-3, 6, 2,-3,-1,-1,-3,-1,-1,-4,-1,-3, 1,-1,-1, 0,-1,-1,-4,-3,-3, 0,-3, 1, 0, 0,
    -1, 1,-4, 2, 5,-3,-2, 0,-3, 0, 1,-3,-2,-3, 1, 0,-1, 0, 0,-1,-4,-2,-2, 0,-3, 4, 0, 0,
    -2,-3,-2,-3,-3, 6,-3,-1, 0,-3,-3, 0,-2, 0,-3,-4,-3,-3,-2,-2,-4,-1, 3, 0,-1,-3, 0, 0,
     0,-1,-3,-1,-2,-3, 6,-2,-4,-2,-2,-4,-3,-2,-2,-2,-2,-3, 0,-2,-4,-3,-2, 0,-3,-2, 0, 0,
    -2, 0,-3,-1, 0,-1,-2, 8,-3,-1,-1,-2,-2, 1,-2,-3, 0,-2,-1,-2,-4,-3,-2, 0, 2, 0, 0, 0,
    -1,-3,-1,-3,-3, 0,-4,-3, 4,-3, 2,-3, 1,-3,-3,-3,-3,-3,-2,-1,-4, 3,-3, 0,-1,-3, 0, 0,
    -1, 0,-3,-1, 0,-3,-2,-1,-3, 4, 2,-2, 0,-2,-1,-2, 0,-2,-1,-1,-4,-1,-1, 0,-1, 2, 0, 0,
    -1, 0,-3,-1, 1,-3,-2,-1, 2, 2, 5,-3, 0,-3, 0,-1,-1,-1,-1,-1,-4,-1,-2, 0,-1, 1, 0, 0,
    -1,-4,-1,-4,-3, 0,-4,-2,-3,-2,-3, 4,-3, 2,-2,-2,-2,-3,-3,-1,-4,-2,-1, 0,-2,-3, 0, 0,
    -1, 0,-1,-1,-2,-2,-3, 1, 1, 0, 0,-3, 5,-2,-1,-2, 0,-2, 0,-1,-4, 1,-2, 0,-1, 0, 0, 0,
    -2, 0,-3,-3,-3, 0,-2, 1,-3,-2,-3, 2,-2, 6,-3,-4,-2,-2,-2,-2,-4,-2,-1, 0, 4,-3, 0, 0,
     0, 3,-3, 1, 1,-3,-2,-2,-3,-1, 0,-2,-1,-3, 7,-1,-2,-1,-1,-1,-4,-2,-3, 0,-3, 1, 0, 0,
    -1,-1,-3,-1, 0,-4,-2,-3,-3,-2,-1,-2,-2,-4,-1, 5, 1,-2, 1,-2,-4,-2,-3, 0,-4,-1, 0, 0,
    -1,-1,-3,-1,-1,-3,-2, 0,-3, 0,-1,-2, 0,-2,-2, 1, 5,-1, 0,-2,-4,-1,-2, 0,-1,-1, 0, 0,
    -1, 0,-3, 0, 0,-3,-3,-2,-3,-2,-2,-3,-2,-2,-1,-2,-1, 5, 0,-2,-4,-2, 2, 0,-3, 0, 0, 0,
     1, 0,-1,-1, 0,-2, 0,-1,-2,-1,-1,-3, 0,-2,-1, 1, 0, 0, 4, 1,-2,-4,-1,-2, 0,-1, 0, 0,
     0,-1,-1,-1,-1,-2,-2,-2,-1,-1,-1,-1,-1,-2,-1,-2,-2,-2, 1, 5,-4, 0,-2, 0,-2,-1, 0, 0,
    -4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4, 1,-4,-4, 0,-4,-4, 0, 0,
    -1,-3,-1,-3,-2,-1,-3,-3, 3,-1,-1,-2, 1,-2,-2,-2,-1,-2,-1, 0,-4, 4,-3, 0,-1,-2, 0, 0,
    -2,-3,-2,-3,-2, 3,-2,-2,-3,-1,-2,-1,-2,-1,-3,-3,-2, 2,-2,-2,-4,-3, 7, 0,-2,-2, 0, 0,
    -1, 0,-1, 0, 0,-2, 0,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0,-1,-1, 0,-1,-1, 0, 0,
    -1,-3,-2,-3,-3,-1,-3, 2,-1,-1,-1,-2,-1, 4,-3,-4,-2,-3,-2,-2,-4,-1,-2, 0, 3,-3, 0, 0,
    -2, 1,-3, 1, 4,-3,-2, 0,-3, 2, 1,-3, 0,-3, 1,-1,-1, 0,-1,-1,-4,-2,-2, 0,-3, 4, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

/* ============================================================
 * Sequence encoding
 * ============================================================ */
unsigned char* diamond_encode_protein(const char* ascii, int len)
{
    unsigned char* out = (unsigned char*)malloc(len);
    if (!out) return NULL;
    for (int i = 0; i < len; i++) {
        out[i] = AA_MAP[(unsigned char)ascii[i]];
    }
    return out;
}

void diamond_fill_blosum62(signed char* matrix)
{
    memcpy(matrix, BLOSUM62, 28*28);
}

/* ============================================================
 * Spaced Seed Shapes (from real Diamond)
 * ============================================================ */

/* Default seed patterns for each sensitivity level */
static const char* SHAPES_DEFAULT[] = {
    "111101110111",
    "111011010010111"
};

static const char* SHAPES_SENSITIVE[] = {
    "1011110111",
    "110100100010111",
    "11001011111",
    "101110001111",
    "11011101100001",
    "1111010010101",
    "111001001001011",
    "10101001101011",
    "111101010011",
    "1111000010000111",
    "1100011011011",
    "1101010000011011",
    "1110001010101001",
    "110011000110011",
    "11011010001101",
    "1101001100010011"
};

static const char* SHAPES_VERY_SENSITIVE[] = {
    "11101111",
    "110110111",
    "111111001",
    "1010111011",
    "11110001011",
    "110100101011",
    "110110001101",
    "1010101000111",
    "1100101001011",
    "1101010101001",
    "1110010010011",
    "110110000010011",
    "111001000100011",
    "1101000100010011"
};

const char** diamond_get_default_shapes(DiamondSensitivity s, int* count)
{
    switch (s) {
        case DIAMOND_SENSITIVITY_DEFAULT:
            *count = sizeof(SHAPES_DEFAULT) / sizeof(SHAPES_DEFAULT[0]);
            return SHAPES_DEFAULT;
        case DIAMOND_SENSITIVITY_SENSITIVE:
        case DIAMOND_SENSITIVITY_MORE_SENSITIVE:
            *count = sizeof(SHAPES_SENSITIVE) / sizeof(SHAPES_SENSITIVE[0]);
            return SHAPES_SENSITIVE;
        case DIAMOND_SENSITIVITY_VERY_SENSITIVE:
        case DIAMOND_SENSITIVITY_ULTRA_SENSITIVE:
            *count = sizeof(SHAPES_VERY_SENSITIVE) / sizeof(SHAPES_VERY_SENSITIVE[0]);
            return SHAPES_VERY_SENSITIVE;
        default:
            *count = 1;
            return SHAPES_DEFAULT;
    }
}

int diamond_seed_shape_init(DiamondSeedShape* shape, const char* pattern)
{
    int len = strlen(pattern);
    if (len > DIAMOND_MAX_SEED_LENGTH) return -1;
    
    shape->weight = 0;
    shape->length = len;
    shape->mask = 0;
    
    for (int i = 0; i < len; i++) {
        if (pattern[i] == '1') {
            if (shape->weight >= DIAMOND_MAX_SEED_WEIGHT) return -1;
            shape->positions[shape->weight] = i;
            shape->mask |= (1U << i);
            shape->weight++;
        }
    }
    
    strcpy(shape->pattern, pattern);
    return shape->weight;
}

/* Hash using spaced seed pattern */
uint64_t diamond_hash_seed_spaced(const unsigned char* seq, 
                                   const DiamondSeedShape* shape)
{
    uint64_t hash = 0;
    for (int i = 0; i < shape->weight; i++) {
        hash *= DIAMOND_ALPHABET_SIZE;
        hash += seq[shape->positions[i]];
    }
    return hash;
}

/* Legacy contiguous 4-mer hash */
uint32_t diamond_hash4(const unsigned char* seq)
{
    return ((uint32_t)seq[0] * (28u * 28u * 28u))
         + ((uint32_t)seq[1] * (28u * 28u))
         + ((uint32_t)seq[2] * 28u)
         + ((uint32_t)seq[3]);
}

/* ============================================================
 * Lookup Table Building
 * ============================================================ */

void diamond_build_lookup_spaced(const unsigned char* query, int qlen,
                                  const DiamondSeedShape* shape,
                                  uint32_t* bucket_count, 
                                  uint32_t* bucket_pos)
{
    /* Clear counts */
    uint64_t hash_space = 1;
    for (int i = 0; i < shape->weight; i++) hash_space *= DIAMOND_ALPHABET_SIZE;
    
    memset(bucket_count, 0, hash_space * sizeof(uint32_t));
    
    /* Build histogram */
    for (int i = 0; i <= qlen - shape->length; i++) {
        uint64_t h = diamond_hash_seed_spaced(query + i, shape);
        if (bucket_count[h] < DIAMOND_MAX_HITS_BUCKET)
            bucket_count[h]++;
    }
    
    /* Prefix sum for positions */
    uint32_t sum = 0, tmp;
    for (uint64_t i = 0; i < hash_space; i++) {
        tmp = bucket_count[i];
        bucket_count[i] = sum;
        sum += tmp;
    }
    
    /* Fill positions */
    memset(bucket_pos, 0, hash_space * DIAMOND_MAX_HITS_BUCKET * sizeof(uint32_t));
    uint32_t* offsets = (uint32_t*)malloc(hash_space * sizeof(uint32_t));
    memcpy(offsets, bucket_count, hash_space * sizeof(uint32_t));
    
    for (int i = 0; i <= qlen - shape->length; i++) {
        uint64_t h = diamond_hash_seed_spaced(query + i, shape);
        uint32_t idx = offsets[h];
        bucket_pos[idx] = i;
        offsets[h]++;
    }
    
    free(offsets);
    
    /* Restore counts */
    for (uint64_t i = hash_space - 1; i > 0; i--) {
        bucket_count[i] = bucket_count[i] - bucket_count[i-1];
    }
}

void diamond_build_lookup(const unsigned char* query, int qlen,
                          uint32_t* bucket_count, uint32_t* bucket_pos)
{
    memset(bucket_count, 0, DIAMOND_HASH_SIZE * sizeof(uint32_t));
    
    /* Build histogram */
    for (int i = 0; i <= qlen - DIAMOND_SEED_SIZE; i++) {
        uint32_t h = diamond_hash4(query + i);
        if (bucket_count[h] < DIAMOND_MAX_HITS_BUCKET)
            bucket_count[h]++;
    }
    
    /* Prefix sum */
    uint32_t sum = 0, tmp;
    for (int i = 0; i < DIAMOND_HASH_SIZE; i++) {
        tmp = bucket_count[i];
        bucket_count[i] = sum;
        sum += tmp;
    }
    
    /* Fill positions */
    memset(bucket_pos, 0, DIAMOND_HASH_SIZE * DIAMOND_MAX_HITS_BUCKET * sizeof(uint32_t));
    uint32_t* offsets = (uint32_t*)malloc(DIAMOND_HASH_SIZE * sizeof(uint32_t));
    memcpy(offsets, bucket_count, DIAMOND_HASH_SIZE * sizeof(uint32_t));
    
    for (int i = 0; i <= qlen - DIAMOND_SEED_SIZE; i++) {
        uint32_t h = diamond_hash4(query + i);
        uint32_t idx = offsets[h];
        bucket_pos[idx] = i;
        offsets[h]++;
    }
    
    free(offsets);
    
    /* Restore counts */
    for (int i = DIAMOND_HASH_SIZE - 1; i > 0; i--) {
        bucket_count[i] = bucket_count[i] - bucket_count[i-1];
    }
}

/* ============================================================
 * Karlin-Altschul Statistics (E-values)
 * ============================================================ */

/* Karlin-Altschul parameters for BLOSUM62 with gap penalties (11,1) */
#define KA_LAMBDA 0.267
#define KA_K      0.041
#define KA_H      0.140

void diamond_init_karlin_altschul(DiamondKarlinAltschul* ka, 
                                   int gap_open, int gap_extend)
{
    /* For BLOSUM62, these are approximately correct */
    ka->lambda = KA_LAMBDA;
    ka->K = KA_K;
    ka->H = KA_H;
    ka->gap_open = gap_open;
    ka->gap_extend = gap_extend;
}

double diamond_bitscore(int score, const DiamondKarlinAltschul* ka)
{
    if (ka->lambda <= 0) return 0.0;
    return (ka->lambda * score - log(ka->K)) / log(2.0);
}

double diamond_evalue(int score, int query_len, 
                       const DiamondDBStats* db_stats,
                       const DiamondKarlinAltschul* ka)
{
    if (ka->lambda <= 0 || ka->K <= 0) return DBL_MAX;
    
    double m = (double)query_len;
    double n = (double)db_stats->db_size;
    double search_space = m * n;
    
    /* Karlin-Altschul: E = K * m * n * e^(-lambda * S) */
    double e = ka->K * search_space * exp(-ka->lambda * score);
    
    return e;
}

/* ============================================================
 * Low-Complexity Masking (Simplified SEG)
 * ============================================================ */
static const float COMPLEXITY_TRIGGER = 2.2f;  /* SEG default */

void diamond_mask_seg(unsigned char* seq, int len, int window, int trigger)
{
    if (window <= 0) window = 12;
    if (trigger <= 0) trigger = (int)(COMPLEXITY_TRIGGER * 100);
    
    for (int i = 0; i <= len - window; i++) {
        int counts[26] = {0};
        for (int j = 0; j < window; j++) {
            unsigned char c = seq[i + j];
            if (c < 20) counts[c]++;
        }
        
        /* Calculate complexity (simplified entropy) */
        float entropy = 0.0f;
        for (int k = 0; k < 20; k++) {
            if (counts[k] > 0) {
                float p = (float)counts[k] / window;
                entropy -= p * logf(p);
            }
        }
        
        /* Mask low complexity regions */
        if (entropy * 100 < trigger) {
            for (int j = 0; j < window; j++) {
                seq[i + j] = 27;  /* Mask character */
            }
        }
    }
}

/* ============================================================
 * Sensitivity Presets
 * ============================================================ */

void diamond_set_sensitivity(DiamondGPUOptions* opts, DiamondSensitivity s)
{
    opts->sensitivity = s;
    
    switch (s) {
        case DIAMOND_SENSITIVITY_FASTER:
            opts->gap_mode = DIAMOND_GAP_NONE;
            opts->x_drop_ungapped = 12;
            opts->min_score = 40;
            opts->band_width = 16;
            break;
            
        case DIAMOND_SENSITIVITY_FAST:
            opts->gap_mode = DIAMOND_GAP_NONE;
            opts->x_drop_ungapped = 16;
            opts->min_score = 35;
            opts->band_width = 16;
            break;
            
        case DIAMOND_SENSITIVITY_DEFAULT:
            opts->gap_mode = DIAMOND_GAP_BANDED_FAST;
            opts->x_drop_ungapped = 16;
            opts->x_drop_gapped = 20;
            opts->min_score = 25;
            opts->band_width = 32;
            opts->gap_open = 11;
            opts->gap_extend = 1;
            break;
            
        case DIAMOND_SENSITIVITY_SENSITIVE:
        case DIAMOND_SENSITIVITY_MORE_SENSITIVE:
            opts->gap_mode = DIAMOND_GAP_BANDED_FAST;
            opts->x_drop_ungapped = 16;
            opts->x_drop_gapped = 20;
            opts->min_score = 20;
            opts->band_width = 48;
            opts->gap_open = 11;
            opts->gap_extend = 1;
            break;
            
        case DIAMOND_SENSITIVITY_VERY_SENSITIVE:
        case DIAMOND_SENSITIVITY_ULTRA_SENSITIVE:
            opts->gap_mode = DIAMOND_GAP_BANDED_SLOW;
            opts->x_drop_ungapped = 20;
            opts->x_drop_gapped = 25;
            opts->min_score = 15;
            opts->band_width = 64;
            opts->gap_open = 11;
            opts->gap_extend = 1;
            break;
    }
}

/* ============================================================
 * HSP Culling and Statistics
 * ============================================================ */

static int hsp_compare(const void* a, const void* b)
{
    const DiamondHSP* h1 = (const DiamondHSP*)a;
    const DiamondHSP* h2 = (const DiamondHSP*)b;
    /* Higher score first */
    return (h2->gapped_score > h1->gapped_score) ? 1 : 
           (h2->gapped_score < h1->gapped_score) ? -1 : 0;
}

void diamond_cull_hsps(DiamondResult* result, int max_hsps)
{
    if (max_hsps <= 0 || result->num_hsps <= max_hsps) return;
    
    /* Sort by gapped score descending */
    qsort(result->hsps, result->num_hsps, sizeof(DiamondHSP), hsp_compare);
    
    /* Keep only top max_hsps */
    result->num_hsps = max_hsps;
    
    /* Recalculate total score */
    result->total_gapped_score = 0;
    for (int i = 0; i < result->num_hsps; i++) {
        result->total_gapped_score += result->hsps[i].gapped_score;
    }
}

float diamond_calc_coverage(int aln_start, int aln_end, int seq_len)
{
    if (seq_len <= 0) return 0.0f;
    int covered = aln_end - aln_start + 1;
    return 100.0f * (float)covered / (float)seq_len;
}

float diamond_calc_pident(const unsigned char* query, int qstart,
                          const unsigned char* subject, int sstart,
                          int aln_len, const signed char* blosum62)
{
    int ident = 0;
    for (int i = 0; i < aln_len; i++) {
        if (query[qstart + i] == subject[sstart + i]) ident++;
    }
    return 100.0f * (float)ident / (float)aln_len;
}

void diamond_compute_stats(DiamondResult* results, int num_results,
                           int query_len,
                           const DiamondGPUOptions* opts)
{
    if (!opts->ka_params || !opts->db_stats) return;
    
    for (int i = 0; i < num_results; i++) {
        DiamondResult* r = &results[i];
        
        for (int j = 0; j < r->num_hsps; j++) {
            DiamondHSP* hsp = &r->hsps[j];
            
            /* Use gapped score if available, else ungapped */
            int score = (hsp->gapped_score > 0) ? hsp->gapped_score : hsp->ungapped_score;
            
            /* Compute statistics */
            hsp->bitscore = diamond_bitscore(score, opts->ka_params);
            hsp->evalue = diamond_evalue(score, query_len, opts->db_stats, opts->ka_params);
            
            /* Coverage */
            if (hsp->gapped_score > 0) {
                hsp->qcov = diamond_calc_coverage(hsp->gapped_q_start, 
                                                  hsp->gapped_q_end, query_len);
                hsp->scov = diamond_calc_coverage(hsp->gapped_s_start,
                                                  hsp->gapped_s_end, r->sseq_len);
            } else {
                hsp->qcov = diamond_calc_coverage(hsp->q_start,
                                                  hsp->q_end, query_len);
                hsp->scov = diamond_calc_coverage(hsp->s_start,
                                                  hsp->s_end, r->sseq_len);
            }
        }
        
        /* Cull if needed */
        if (opts->max_hsps_per_target > 0) {
            diamond_cull_hsps(r, opts->max_hsps_per_target);
        }
    }
}

/* ============================================================
 * Output Functions
 * ============================================================ */

void diamond_output_header(DiamondOutputFormat fmt, int fields)
{
    (void)fields;
    
    switch (fmt) {
        case DIAMOND_OUTPUT_TABULAR:
            printf("# Fields: query id, subject id, %% identity, alignment length, ");
            printf("mismatches, gap opens, q. start, q. end, s. start, s. end, ");
            printf("evalue, bit score\n");
            break;
        case DIAMOND_OUTPUT_SAM:
            printf("@HD\tVN:1.0\tSO:unsorted\n");
            break;
        default:
            break;
    }
}

void diamond_output_result(const DiamondResult* result, int result_idx,
                          const char* qname, int qlen,
                          DiamondOutputFormat fmt, int fields)
{
    (void)fields;
    const char* sname = result->sseq_name[0] ? result->sseq_name : "unnamed";
    
    for (int i = 0; i < result->num_hsps; i++) {
        const DiamondHSP* h = &result->hsps[i];
        
        switch (fmt) {
            case DIAMOND_OUTPUT_TEXT:
                printf("Result %d [%s]: HSP %d: score=%d (ungapped=%d), "
                       "evalue=%.2e, bitscore=%.1f, "
                       "Q[%d-%d] S[%d-%d], qcov=%.1f%%, scov=%.1f%%\n",
                       result_idx, sname, i,
                       h->gapped_score ? h->gapped_score : h->ungapped_score,
                       h->ungapped_score,
                       h->evalue, h->bitscore,
                       h->gapped_q_start, h->gapped_q_end,
                       h->gapped_s_start, h->gapped_s_end,
                       h->qcov, h->scov);
                break;
                
            case DIAMOND_OUTPUT_TABULAR:
                /* BLAST tabular format */
                printf("%s\t%s\t%.2f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.2e\t%.1f\n",
                       qname, sname,
                       h->pident > 0 ? h->pident : 0.0f,
                       h->length > 0 ? h->length : 
                           (h->gapped_q_end - h->gapped_q_start + 1),
                       (h->length > 0 ? h->length : 0) - h->nident,
                       h->ngaps,
                       h->gapped_q_start + 1,  /* 1-based */
                       h->gapped_q_end + 1,
                       h->gapped_s_start + 1,
                       h->gapped_s_end + 1,
                       h->evalue,
                       h->bitscore);
                break;
                
            case DIAMOND_OUTPUT_SAM:
                /* Simplified SAM output */
                printf("%s\t0\t%s\t%d\t255\t%dM\t*\t0\t0\t*\t*\tAS:i:%d\t"
                       "EV:f:%.2e\tBS:f:%.1f\n",
                       qname, sname,
                       h->gapped_s_start + 1,
                       h->gapped_q_end - h->gapped_q_start + 1,
                       h->gapped_score,
                       h->evalue,
                       h->bitscore);
                break;
                
            case DIAMOND_OUTPUT_PAF:
                /* PAF format */
                printf("%s\t%d\t%d\t%d\t+\t%s\t%d\t%d\t%d\t%d\t%d\t%.0f\n",
                       qname, qlen,
                       h->gapped_q_start,
                       h->gapped_q_end + 1,
                       sname, result->sseq_len,
                       h->gapped_s_start,
                       h->gapped_s_end + 1,
                       h->nident,
                       h->length > 0 ? h->length : (h->gapped_q_end - h->gapped_q_start + 1),
                       h->evalue < 1e-300 ? 0 : -log10(h->evalue) * 10);
                break;
                
            default:
                break;
        }
    }
}

void diamond_output_footer(DiamondOutputFormat fmt)
{
    (void)fmt;
    /* No footer needed for most formats */
}
