#include "gpu_diamond.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* NCBIstdaa-style mapping for 28-letter alphabet, matching the
 * BLOSUM62 layout used by NCBI BLAST and the original gpu_blastp.c.
 * Index 0 = gap, 1..23 = amino acids, 24..27 = ambiguity/stop. */
static int aa_code(char c)
{
    switch (c) {
        case '-': return 0;
        case 'A': return 1;  case 'B': return 2;  case 'C': return 3;
        case 'D': return 4;  case 'E': return 5;  case 'F': return 6;
        case 'G': return 7;  case 'H': return 8;  case 'I': return 9;
        case 'K': return 10; case 'L': return 11; case 'M': return 12;
        case 'N': return 13; case 'P': return 14; case 'Q': return 15;
        case 'R': return 16; case 'S': return 17; case 'T': return 18;
        case 'V': return 19; case 'W': return 20; case 'X': return 21;
        case 'Y': return 22; case 'Z': return 23;
        case 'U': return 24; case '*': return 25; case 'O': return 26;
        case 'J': return 27;
        default:  return 21; /* X */
    }
}

unsigned char* diamond_encode_protein(const char* ascii, int len)
{
    unsigned char* out = (unsigned char*)malloc(len);
    if (!out) return NULL;
    for (int i = 0; i < len; ++i) {
        char c = ascii[i];
        if (c >= 'a' && c <= 'z') c = (char)(c - 'a' + 'A');
        out[i] = (unsigned char)aa_code(c);
    }
    return out;
}

/* BLOSUM62 in 28x28 layout, identical to the matrix used in gpu_blastp.c */
static const signed char BLOSUM62[28][28] = {
    {-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128,-128},
    {-128,   4,  -2,   0,  -2,  -1,  -2,   0,  -2,  -1,  -1,  -1,  -1,  -2,  -1,  -1,  -1,   1,   0,   0,  -3,  -1,  -2,  -1,  -1,  -4,  -1,  -1},
    {-128,  -2,   4,  -3,   4,   1,  -3,  -1,   0,  -3,   0,  -4,  -3,   4,  -2,   0,  -1,   0,  -1,  -3,  -4,  -1,  -3,   0,  -1,  -4,  -1,  -3},
    {-128,   0,  -3,   9,  -3,  -4,  -2,  -3,  -3,  -1,  -3,  -1,  -1,  -3,  -3,  -3,  -3,  -1,  -1,  -1,  -2,  -1,  -2,  -3,  -1,  -4,  -1,  -1},
    {-128,  -2,   4,  -3,   6,   2,  -3,  -1,  -1,  -3,  -1,  -4,  -3,   1,  -1,   0,  -2,   0,  -1,  -3,  -4,  -1,  -3,   1,  -1,  -4,  -1,  -3},
    {-128,  -1,   1,  -4,   2,   5,  -3,  -2,   0,  -3,   1,  -3,  -2,   0,  -1,   2,   0,   0,  -1,  -2,  -3,  -1,  -2,   4,  -1,  -4,  -1,  -3},
    {-128,  -2,  -3,  -2,  -3,  -3,   6,  -3,  -1,   0,  -3,   0,   0,  -3,  -4,  -3,  -3,  -2,  -2,  -1,   1,  -1,   3,  -3,  -1,  -4,  -1,   0},
    {-128,   0,  -1,  -3,  -1,  -2,  -3,   6,  -2,  -4,  -2,  -4,  -3,   0,  -2,  -2,  -2,   0,  -2,  -3,  -2,  -1,  -3,  -2,  -1,  -4,  -1,  -4},
    {-128,  -2,   0,  -3,  -1,   0,  -1,  -2,   8,  -3,  -1,  -3,  -2,   1,  -2,   0,   0,  -1,  -2,  -3,  -2,  -1,   2,   0,  -1,  -4,  -1,  -3},
    {-128,  -1,  -3,  -1,  -3,  -3,   0,  -4,  -3,   4,  -3,   2,   1,  -3,  -3,  -3,  -3,  -2,  -1,   3,  -3,  -1,  -1,  -3,  -1,  -4,  -1,   3},
    {-128,  -1,   0,  -3,  -1,   1,  -3,  -2,  -1,  -3,   5,  -2,  -1,   0,  -1,   1,   2,   0,  -1,  -2,  -3,  -1,  -2,   1,  -1,  -4,  -1,  -3},
    {-128,  -1,  -4,  -1,  -4,  -3,   0,  -4,  -3,   2,  -2,   4,   2,  -3,  -3,  -2,  -2,  -2,  -1,   1,  -2,  -1,  -1,  -3,  -1,  -4,  -1,   3},
    {-128,  -1,  -3,  -1,  -3,  -2,   0,  -3,  -2,   1,  -1,   2,   5,  -2,  -2,   0,  -1,  -1,  -1,   1,  -1,  -1,  -1,  -1,  -1,  -4,  -1,   2},
    {-128,  -2,   4,  -3,   1,   0,  -3,   0,   1,  -3,   0,  -3,  -2,   6,  -2,   0,   0,   1,   0,  -3,  -4,  -1,  -2,   0,  -1,  -4,  -1,  -3},
    {-128,  -1,  -2,  -3,  -1,  -1,  -4,  -2,  -2,  -3,  -1,  -3,  -2,  -2,   7,  -1,  -2,  -1,  -1,  -2,  -4,  -1,  -3,  -1,  -1,  -4,  -1,  -3},
    {-128,  -1,   0,  -3,   0,   2,  -3,  -2,   0,  -3,   1,  -2,   0,   0,  -1,   5,   1,   0,  -1,  -2,  -2,  -1,  -1,   4,  -1,  -4,  -1,  -2},
    {-128,  -1,  -1,  -3,  -2,   0,  -3,  -2,   0,  -3,   2,  -2,  -1,   0,  -2,   1,   5,  -1,  -1,  -3,  -3,  -1,  -2,   0,  -1,  -4,  -1,  -2},
    {-128,   1,   0,  -1,   0,   0,  -2,   0,  -1,  -2,   0,  -2,  -1,   1,  -1,   0,  -1,   4,   1,  -2,  -3,  -1,  -2,   0,  -1,  -4,  -1,  -2},
    {-128,   0,  -1,  -1,  -1,  -1,  -2,  -2,  -2,  -1,  -1,  -1,  -1,   0,  -1,  -1,  -1,   1,   5,   0,  -2,  -1,  -2,  -1,  -1,  -4,  -1,  -1},
    {-128,   0,  -3,  -1,  -3,  -2,  -1,  -3,  -3,   3,  -2,   1,   1,  -3,  -2,  -2,  -3,  -2,   0,   4,  -3,  -1,  -1,  -2,  -1,  -4,  -1,   2},
    {-128,  -3,  -4,  -2,  -4,  -3,   1,  -2,  -2,  -3,  -3,  -2,  -1,  -4,  -4,  -2,  -3,  -3,  -2,  -3,  11,  -1,   2,  -2,  -1,  -4,  -1,  -2},
    {-128,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -4,  -1,  -1},
    {-128,  -2,  -3,  -2,  -3,  -2,   3,  -3,   2,  -1,  -2,  -1,  -1,  -2,  -3,  -1,  -2,  -2,  -2,  -1,   2,  -1,   7,  -2,  -1,  -4,  -1,  -1},
    {-128,  -1,   0,  -3,   1,   4,  -3,  -2,   0,  -3,   1,  -3,  -1,   0,  -1,   4,   0,   0,  -1,  -2,  -2,  -1,  -2,   4,  -1,  -4,  -1,  -3},
    {-128,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -4,  -1,  -1},
    {-128,  -4,  -4,  -4,  -4,  -4,  -4,  -4,  -4,  -4,  -4,  -4,  -4,  -4,  -4,  -4,  -4,  -4,  -4,  -4,  -4,  -4,  -4,  -4,  -4,   1,  -4,  -4},
    {-128,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -4,  -1,  -1},
    {-128,  -1,  -3,  -1,  -3,  -3,   0,  -4,  -3,   3,  -3,   3,   2,  -3,  -3,  -2,  -2,  -2,  -1,   2,  -2,  -1,  -1,  -3,  -1,  -4,  -1,   3}
};

void diamond_fill_blosum62(signed char* matrix)
{
    memcpy(matrix, BLOSUM62, sizeof(BLOSUM62));
}

uint32_t diamond_hash4(const unsigned char* seq)
{
    /* Polynomial hash: a0*28^3 + a1*28^2 + a2*28 + a3 */
    return ((uint32_t)seq[0] * (28u * 28u * 28u))
         + ((uint32_t)seq[1] * (28u * 28u))
         + ((uint32_t)seq[2] * 28u)
         + ((uint32_t)seq[3]);
}

void diamond_build_lookup(const unsigned char* query, int qlen,
                          uint32_t* bucket_count, uint32_t* bucket_pos)
{
    memset(bucket_count, 0, DIAMOND_HASH_SIZE * sizeof(uint32_t));

    if (qlen < DIAMOND_SEED_SIZE) return;
    for (int i = 0; i + DIAMOND_SEED_SIZE <= qlen; ++i) {
        uint32_t h = diamond_hash4(query + i);
        if (h >= DIAMOND_HASH_SIZE) continue; /* defensive */
        uint32_t c = bucket_count[h];
        if (c < DIAMOND_MAX_HITS_BUCKET) {
            bucket_pos[h * DIAMOND_MAX_HITS_BUCKET + c] = (uint32_t)i;
            bucket_count[h] = c + 1;
        }
    }
}
