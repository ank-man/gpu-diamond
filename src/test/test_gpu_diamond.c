/* Standalone test harness for GPU-Diamond. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "gpu_diamond.h"

#define QUERY_SEQ \
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKAL" \
    "PDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMG"

static const char* SUBJECTS[] = {
    /* perfect match */
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKAL"
    "PDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMG",
    /* partial overlap (first ~50 aa match) */
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGZZZZZZZZZZZZ",
    /* unrelated */
    "MTMDKSELVQKAKLAEQAERYDEMVEYRRSVQVGETSTGELNKVSATEHASRTLKSTTEEKF",
    /* short, no hit possible */
    "AAA",
};
static const char* NAMES[] = {
    "perfect_match",
    "partial_match",
    "unrelated",
    "too_short",
};
#define NSEQ ((int)(sizeof(SUBJECTS)/sizeof(SUBJECTS[0])))

int main(void)
{
    printf("GPU-Diamond standalone test\n");
    printf("===========================\n\n");

    /* ----- Encode query ----- */
    int qlen = (int)strlen(QUERY_SEQ);
    unsigned char* q_enc = diamond_encode_protein(QUERY_SEQ, qlen);
    printf("Query length: %d aa\n", qlen);

    /* ----- Build padded database ----- */
    int lens[NSEQ];
    int max_len = 0;
    for (int i = 0; i < NSEQ; ++i) {
        lens[i] = (int)strlen(SUBJECTS[i]);
        if (lens[i] > max_len) max_len = lens[i];
    }
    int padded = ((max_len + 3) / 4) * 4;  /* multiple of 4 */
    if (padded < DIAMOND_SEED_SIZE) padded = DIAMOND_SEED_SIZE;

    unsigned char* db = (unsigned char*)calloc((size_t)NSEQ * padded, 1);
    for (int i = 0; i < NSEQ; ++i) {
        unsigned char* enc = diamond_encode_protein(SUBJECTS[i], lens[i]);
        memcpy(db + (size_t)i * padded, enc, lens[i]);
        free(enc);
    }
    printf("Database: %d sequences, max=%d aa, padded=%d\n\n", NSEQ, max_len, padded);

    /* ----- Configure search ----- */
    DiamondGPUOptions opts;
    opts.num_blocks     = 1;
    opts.num_threads    = NSEQ;     /* one thread per sequence is plenty for the test */
    opts.num_sequences  = NSEQ;
    opts.padded_length  = padded;
    opts.x_drop         = 16;
    opts.min_score      = 25;
    opts.seed_score_min = 12;

    DiamondResult* results = (DiamondResult*)calloc(NSEQ, sizeof(DiamondResult));

    clock_t t0 = clock();
    int rc = diamond_gpu_search(q_enc, qlen, db, lens, &opts, results);
    clock_t t1 = clock();

    if (rc != 0) {
        fprintf(stderr, "GPU search failed (rc=%d)\n", rc);
        free(q_enc); free(db); free(results);
        return 1;
    }

    /* ----- Print results ----- */
    printf("Elapsed: %.3f ms\n\n", 1000.0 * (double)(t1 - t0) / CLOCKS_PER_SEC);
    int total_hsps = 0;
    for (int i = 0; i < NSEQ; ++i) {
        printf("[%d] %-15s  seed_hits=%-5d ext=%-3d hsps=%d  total_score=%d\n",
               i, NAMES[i], results[i].seed_hits, results[i].extensions,
               results[i].num_hsps, results[i].total_score);
        for (int j = 0; j < results[i].num_hsps; ++j) {
            DiamondHSP* h = &results[i].hsps[j];
            printf("     hsp%d: q[%d..%d] s[%d..%d] score=%d\n",
                   j, h->q_start, h->q_end, h->s_start, h->s_end, h->score);
        }
        total_hsps += results[i].num_hsps;
    }
    printf("\nTotal HSPs: %d\n", total_hsps);

    /* ----- Sanity check: perfect match should produce a high-scoring HSP ----- */
    int ok = 1;
    if (results[0].num_hsps == 0 || results[0].hsps[0].score < qlen) {
        printf("WARNING: perfect_match should produce a near-full-length HSP\n");
        ok = 0;
    }
    if (results[3].num_hsps != 0) {
        printf("WARNING: too_short should produce no HSPs\n");
        ok = 0;
    }
    printf("\n%s\n", ok ? "TEST PASSED" : "TEST FAILED (see warnings)");

    free(q_enc); free(db); free(results);
    return ok ? 0 : 2;
}
