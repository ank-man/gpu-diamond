/****
GPU-Diamond ungapped extension implementation
****/
#include "ungapped.h"
#include <stddef.h>

extern const signed char BLOSUM62[28*28];

void ungapped_extend(const Sequence* query, int q_anchor,
                     const Sequence* subject, int s_anchor,
                     int xdrop, const signed char* score_matrix,
                     UngappedResult* result)
{
    int score = 0, st = 0, n = 1;
    int delta = 0;
    
    /* Extend left */
    int q = q_anchor - 1;
    int s = s_anchor - 1;
    int best_left = 0;
    
    while (score - st < xdrop && q >= 0 && s >= 0) {
        st += score_matrix[query->data[q] * 28 + subject->data[s]];
        if (st > score) {
            score = st;
            delta = n;
        }
        q--; s--; n++;
    }
    result->left_score = score;
    int left_q = q_anchor - delta;
    int left_s = s_anchor - delta;
    
    /* Extend right */
    score = 0; st = 0; n = 1;
    int len_right = 0;
    q = q_anchor;
    s = s_anchor;
    int best_right = 0;
    
    while (score - st < xdrop && q < query->length && s < subject->length) {
        st += score_matrix[query->data[q] * 28 + subject->data[s]];
        if (st > score) {
            score = st;
            len_right = n;
        }
        q++; s++; n++;
    }
    result->right_score = score;
    
    /* Calculate total */
    int seed_score = score_matrix[query->data[q_anchor] * 28 + subject->data[s_anchor]];
    result->score = result->left_score + seed_score + result->right_score;
    
    /* Set coordinates */
    result->q_start = left_q;
    result->q_end = q_anchor + len_right - 1;
    result->s_start = left_s;
    result->s_end = s_anchor + len_right - 1;
    result->length = (result->q_end - result->q_start + 1);
}

int score_seed(const unsigned char* query, int qpos,
               const unsigned char* subject, int spos,
               int seed_len, const signed char* score_matrix)
{
    int score = 0;
    for (int k = 0; k < seed_len; k++) {
        score += score_matrix[query[qpos + k] * 28 + subject[spos + k]];
    }
    return score;
}
