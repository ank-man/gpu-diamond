/****
GPU-Diamond sequence data structures implementation
****/
#include "sequence.h"
#include <stdlib.h>
#include <string.h>

extern unsigned char* diamond_encode_protein(const char* ascii, int len);

Block* block_create(const char** sequences, int num_seqs)
{
    Block* block = calloc(1, sizeof(Block));
    if (!block) return NULL;
    
    block->num_seqs = num_seqs;
    block->lengths = calloc(num_seqs, sizeof(int));
    block->offsets = calloc(num_seqs, sizeof(int));
    
    /* Find max length */
    int max_len = 0;
    for (int i = 0; i < num_seqs; i++) {
        int len = strlen(sequences[i]);
        block->lengths[i] = len;
        if (len > max_len) max_len = len;
    }
    
    block->max_len = max_len;
    block->padded_len = max_len + 4;  /* Small padding */
    
    /* Allocate padded data */
    block->data = calloc((size_t)num_seqs * block->padded_len, 1);
    
    /* Encode and copy sequences */
    for (int i = 0; i < num_seqs; i++) {
        block->offsets[i] = i * block->padded_len;
        unsigned char* enc = diamond_encode_protein(sequences[i], block->lengths[i]);
        memcpy(block->data + block->offsets[i], enc, block->lengths[i]);
        free(enc);
    }
    
    return block;
}

void block_free(Block* block)
{
    if (!block) return;
    free(block->data);
    free(block->lengths);
    free(block->offsets);
    free(block);
}

Sequence block_get_sequence(const Block* block, int idx)
{
    Sequence seq = {
        .data = block->data + block->offsets[idx],
        .length = block->lengths[idx],
        .id = idx
    };
    return seq;
}
