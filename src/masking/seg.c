/****
GPU-Diamond SEG masking implementation (simplified)
****/
#include "seg.h"
#include <math.h>

double seg_entropy(const int* counts, int alphabet_size)
{
    double entropy = 0.0;
    int total = 0;
    
    for (int i = 0; i < alphabet_size; i++) total += counts[i];
    if (total == 0) return 0.0;
    
    for (int i = 0; i < alphabet_size; i++) {
        if (counts[i] > 0) {
            double p = (double)counts[i] / total;
            entropy -= p * log(p);
        }
    }
    return entropy;
}

void seg_mask(unsigned char* seq, int len, const SEGParams* params)
{
    int window = params->window > 0 ? params->window : SEG_WINDOW;
    double trigger = params->trigger > 0 ? params->trigger : SEG_TRIGGER;
    
    /* Simple sliding window SEG */
    for (int i = 0; i <= len - window; i++) {
        int counts[28] = {0};
        
        /* Count residues in window */
        for (int j = 0; j < window; j++) {
            unsigned char c = seq[i + j];
            if (c < 28) counts[c]++;
        }
        
        /* Check entropy */
        double entropy = seg_entropy(counts, 20);  /* First 20 are amino acids */
        
        /* Mask if below trigger */
        if (entropy < trigger) {
            for (int j = 0; j < window; j++) {
                seq[i + j] = 27;  /* Mask character */
            }
        }
    }
}
