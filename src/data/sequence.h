/****
GPU-Diamond protein sequence aligner
Minimal sequence data structures (like real Diamond)
****/

#ifndef SEQUENCE_H
#define SEQUENCE_H

#include <stdint.h>

/* Sequence representation (like Diamond's Sequence class) */
typedef struct {
    const unsigned char* data;  /* Encoded sequence data */
    int length;                 /* Sequence length */
    int id;                     /* Sequence ID */
} Sequence;

/* Block of sequences (like Diamond's Block) */
typedef struct {
    unsigned char* data;        /* Padded sequence data */
    int* lengths;               /* Array of sequence lengths */
    int* offsets;               /* Offset of each sequence in data */
    int num_seqs;               /* Number of sequences */
    int max_len;                /* Max sequence length */
    int padded_len;             /* Padded length per sequence */
} Block;

/* Create a sequence block from raw sequences */
Block* block_create(const char** sequences, int num_seqs);
void block_free(Block* block);

/* Get sequence from block */
Sequence block_get_sequence(const Block* block, int idx);

#endif
