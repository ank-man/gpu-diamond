/****
GPU-Diamond Karlin-Altschul statistics
Like real Diamond's stats/score_matrix.h and stats functions
****/

#ifndef KARLIN_ALTSCHUL_H
#define KARLIN_ALTSCHUL_H

/* Karlin-Altschul parameters for BLOSUM62 with gap (11,1) */
typedef struct {
    double lambda;      /* Karlin-Altschul lambda */
    double K;           /* Karlin-Altschul K */
    double H;           /* Relative entropy */
    int gap_open;       /* Gap open penalty */
    int gap_extend;     /* Gap extend penalty */
} KarlinAltschul;

/* Database statistics */
typedef struct {
    int64_t db_size;        /* Total database residues */
    int db_seqs;            /* Number of sequences */
    double search_space;    /* Effective search space */
} DBStats;

/* Initialize with BLOSUM62 defaults */
void ka_init_blosum62(KarlinAltschul* ka, int gap_open, int gap_extend);

/* Compute bit score from raw score */
double ka_bitscore(int score, const KarlinAltschul* ka);

/* Compute E-value */
double ka_evalue(int score, int query_len, const DBStats* db, const KarlinAltschul* ka);

#endif
