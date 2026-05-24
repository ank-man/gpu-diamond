/****
GPU-Diamond ungapped X-drop extension
Like real Diamond's dp/ungapped.h and dp/ungapped_align.cpp
****/

#ifndef UNGAPPED_H
#define UNGAPPED_H

#include "../data/sequence.h"

/* Ungapped extension result */
typedef struct {
    int score;          /* Total extension score */
    int left_score;     /* Left extension score */
    int right_score;    /* Right extension score */
    int q_start;        /* Query start position */
    int q_end;          /* Query end position */
    int s_start;        /* Subject start position */
    int s_end;          /* Subject end position */
    int length;         /* Alignment length */
} UngappedResult;

/* X-drop ungapped extension (like diamond xdrop_ungapped) */
void ungapped_extend(const Sequence* query, int q_anchor,
                     const Sequence* subject, int s_anchor,
                     int xdrop, const signed char* score_matrix,
                     UngappedResult* result);

/* Score a seed match */
int score_seed(const unsigned char* query, int qpos,
               const unsigned char* subject, int spos,
               int seed_len, const signed char* score_matrix);

#endif
