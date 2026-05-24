/****
GPU-Diamond SEG low-complexity masking
Like real Diamond's masking/def.h and masking functions
****/

#ifndef SEG_H
#define SEG_H

/* SEG-style low-complexity masking parameters */
typedef struct {
    int window;         /* Window size for complexity check */
    double trigger;     /* Complexity trigger (entropy threshold) */
    int extension;      /* Extension length for masked regions */
} SEGParams;

/* Default SEG parameters */
#define SEG_WINDOW 12
#define SEG_TRIGGER 2.2
#define SEG_EXTENSION 1

/* Mask low-complexity regions in sequence
 * Replaces masked positions with 27 (mask character)
 */
void seg_mask(unsigned char* seq, int len, const SEGParams* params);

/* Simple entropy calculation for a window */
double seg_entropy(const int* counts, int alphabet_size);

#endif
