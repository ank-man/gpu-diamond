/****
GPU-Diamond Karlin-Altschul statistics implementation
****/
#include "karlin_altschul.h"
#include <math.h>

/* Karlin-Altschul parameters for BLOSUM62 with gap (11,1) */
#define KA_LAMBDA 0.267
#define KA_K      0.041
#define KA_H      0.140

void ka_init_blosum62(KarlinAltschul* ka, int gap_open, int gap_extend)
{
    ka->lambda = KA_LAMBDA;
    ka->K = KA_K;
    ka->H = KA_H;
    ka->gap_open = gap_open;
    ka->gap_extend = gap_extend;
}

double ka_bitscore(int score, const KarlinAltschul* ka)
{
    if (ka->lambda <= 0) return 0.0;
    return (ka->lambda * score - log(ka->K)) / log(2.0);
}

double ka_evalue(int score, int query_len, const DBStats* db, const KarlinAltschul* ka)
{
    if (ka->lambda <= 0 || ka->K <= 0) return 1e300;
    
    double m = (double)query_len;
    double n = (double)db->db_size;
    
    /* Karlin-Altschul: E = K * m * n * e^(-lambda * S) */
    return ka->K * m * n * exp(-ka->lambda * score);
}
