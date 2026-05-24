/* test_gpu_diamond_v2.c - Test program for enhanced GPU-Diamond */
#include "gpu_diamond_v2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* Test sequences */
static const char* QUERY_SEQ = 
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLG";

static const char* TEST_SEQUENCES[] = {
    /* perfect_match: exact match of query */
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLG",
    /* partial_match: 80% similar */
    "MKTAYIAKQRLISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLG",
    /* unrelated: random protein */
    "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYAC",
    /* too_short: only 10 AA */
    "MKTAYIAKQR",
    /* gapped_match: similar but with gaps */
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRV---GTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLG",
};

static const char* SEQ_NAMES[] = {
    "perfect_match",
    "partial_match",
    "unrelated",
    "too_short",
    "gapped_match"
};

#define NUM_SEQ (sizeof(TEST_SEQUENCES)/sizeof(TEST_SEQUENCES[0]))

/* Helper: find max length */
static int max_len(const char** seqs, int n)
{
    int m = 0;
    for (int i = 0; i < n; i++) {
        int l = strlen(seqs[i]);
        if (l > m) m = l;
    }
    return m;
}

/* Encode and pad database */
static unsigned char* prepare_db(const char** seqs, int n, int padded_len, int* out_lengths)
{
    unsigned char* db = calloc((size_t)n * padded_len, 1);
    if (!db) return NULL;
    
    for (int i = 0; i < n; i++) {
        int len = strlen(seqs[i]);
        out_lengths[i] = len;
        unsigned char* enc = diamond_encode_protein(seqs[i], len);
        memcpy(db + (size_t)i * padded_len, enc, len);
        free(enc);
    }
    return db;
}

/* Print HSP details */
static void print_hsp(const DiamondHSP* h, int idx)
{
    printf("  HSP %d: ungapped_score=%d", idx, h->ungapped_score);
    if (h->gapped_score > 0) {
        printf(" gapped_score=%d", h->gapped_score);
    }
    printf("\n");
    printf("    Query:  [%d - %d] -> [%d - %d] (gapped)\n",
           h->q_start + 1, h->q_end + 1,
           h->gapped_q_start + 1, h->gapped_q_end + 1);
    printf("    Sbjct:  [%d - %d] -> [%d - %d] (gapped)\n",
           h->s_start + 1, h->s_end + 1,
           h->gapped_s_start + 1, h->gapped_s_end + 1);
    printf("    Stats:  evalue=%.2e bitscore=%.1f pident=%.1f%%\n",
           h->evalue, h->bitscore, h->pident);
    printf("    Coverage: qcov=%.1f%% scov=%.1f%%\n", h->qcov, h->scov);
}

int main(int argc, char** argv)
{
    (void)argc; (void)argv;
    
    printf("GPU-Diamond v2.0 Enhanced Test\n");
    printf("================================\n\n");
    
    /* Encode query */
    int qlen = strlen(QUERY_SEQ);
    unsigned char* query = diamond_encode_protein(QUERY_SEQ, qlen);
    printf("Query: %d aa\n", qlen);
    
    /* Prepare database */
    int seq_lengths[NUM_SEQ];
    int max_seq_len = max_len(TEST_SEQUENCES, NUM_SEQ);
    int padded = max_seq_len + 4;  /* Small padding for test */
    
    unsigned char* db = prepare_db(TEST_SEQUENCES, NUM_SEQ, padded, seq_lengths);
    printf("Database: %zu sequences, max=%d, padded=%d\n\n", NUM_SEQ, max_seq_len, padded);
    
    /* ===== Test 1: Contiguous 4-mer seeds (DEFAULT) ===== */
    printf("--- Test 1: Contiguous 4-mer (DEFAULT sensitivity) ---\n");
    
    /* Build lookup for contiguous 4-mers */
    uint32_t* bucket_count = calloc(614656, sizeof(uint32_t));
    uint32_t* bucket_pos = calloc(614656 * 32, sizeof(uint32_t));
    diamond_build_lookup(query, qlen, bucket_count, bucket_pos);
    
    /* Run search - TODO: integrate with CUDA kernel */
    printf("Lookup table built: %d hash buckets\n", 614656);
    printf("Query has ~%d 4-mers\n", qlen - 3);
    
    /* Print sample hash lookups */
    printf("Sample seed hashes:\n");
    for (int i = 0; i < 5 && i < qlen - 3; i++) {
        uint32_t h = diamond_hash4(query + i);
        printf("  Pos %d: hash=%u, hits=%u\n", i, h, bucket_count[h]);
    }
    
    free(bucket_count);
    free(bucket_pos);
    
    /* ===== Test 2: Spaced Seeds ===== */
    printf("\n--- Test 2: Spaced Seeds ---\n");
    
    DiamondSeedShape shape;
    const char* pattern = "111101110111";  /* Diamond default shape */
    int w = diamond_seed_shape_init(&shape, pattern);
    printf("Shape '%s': weight=%d, length=%d\n", pattern, w, shape.length);
    
    /* Get default shapes for sensitivity levels */
    int count;
    const char** shapes;
    
    shapes = diamond_get_default_shapes(DIAMOND_SENSITIVITY_DEFAULT, &count);
    printf("DEFAULT mode uses %d shapes:\n", count);
    for (int i = 0; i < count && i < 4; i++) {
        printf("  %d: %s\n", i, shapes[i]);
    }
    
    shapes = diamond_get_default_shapes(DIAMOND_SENSITIVITY_SENSITIVE, &count);
    printf("SENSITIVE mode uses %d shapes\n", count);
    
    /* Test spaced seed hashing */
    uint64_t h_spaced = diamond_hash_seed_spaced(query, &shape);
    printf("First spaced seed hash: %lu\n", (unsigned long)h_spaced);
    
    /* ===== Test 3: Karlin-Altschul Statistics ===== */
    printf("\n--- Test 3: Karlin-Altschul Statistics ---\n");
    
    DiamondKarlinAltschul ka;
    diamond_init_karlin_altschul(&ka, 11, 1);
    printf("BLOSUM62 (11,1): lambda=%.3f K=%.3f H=%.3f\n",
           ka.lambda, ka.K, ka.H);
    
    /* Test bitscore/evalue calculations */
    int test_scores[] = {50, 100, 200, 400};
    DiamondDBStats db_stats = {
        .db_size = 1000000,  /* 1M residues */
        .db_seqs = 5000,
        .search_space = 0
    };
    db_stats.search_space = (double)qlen * db_stats.db_size;
    
    printf("\nScore -> Bitscore / E-value (vs 1M residue db):\n");
    for (int i = 0; i < 4; i++) {
        int sc = test_scores[i];
        double bits = diamond_bitscore(sc, &ka);
        double e = diamond_evalue(sc, qlen, &db_stats, &ka);
        printf("  %3d -> %.1f bits, %.2e evalue\n", sc, bits, e);
    }
    
    /* ===== Test 4: Sensitivity Presets ===== */
    printf("\n--- Test 4: Sensitivity Presets ---\n");
    
    DiamondGPUOptions opts;
    memset(&opts, 0, sizeof(opts));
    
    DiamondSensitivity modes[] = {
        DIAMOND_SENSITIVITY_FASTER,
        DIAMOND_SENSITIVITY_FAST,
        DIAMOND_SENSITIVITY_DEFAULT,
        DIAMOND_SENSITIVITY_SENSITIVE,
        DIAMOND_SENSITIVITY_MORE_SENSITIVE
    };
    const char* mode_names[] = {"FASTER", "FAST", "DEFAULT", "SENSITIVE", "MORE_SENSITIVE"};
    
    for (int i = 0; i < 5; i++) {
        diamond_set_sensitivity(&opts, modes[i]);
        printf("%s: gap_mode=%d xdrop_ungapped=%d min_score=%d band_width=%d\n",
               mode_names[i], opts.gap_mode, opts.x_drop_ungapped,
               opts.min_score, opts.band_width);
    }
    
    /* ===== Test 5: Output Formats ===== */
    printf("\n--- Test 5: Output Formats ---\n");
    
    /* Create dummy result for output testing */
    DiamondResult result;
    memset(&result, 0, sizeof(result));
    strcpy(result.sseq_name, "test_subject");
    result.sseq_len = 100;
    result.num_hsps = 1;
    
    DiamondHSP* h = &result.hsps[0];
    h->ungapped_score = 150;
    h->gapped_score = 180;
    h->q_start = 10; h->q_end = 40;
    h->s_start = 5; h->s_end = 35;
    h->gapped_q_start = 8; h->gapped_q_end = 42;
    h->gapped_s_start = 3; h->gapped_s_end = 37;
    h->evalue = 1e-25;
    h->bitscore = 45.2;
    h->pident = 85.7f;
    h->length = 35;
    h->nident = 30;
    h->ngaps = 2;
    h->qcov = 28.5f;
    h->scov = 35.0f;
    
    printf("\nTEXT format:\n");
    diamond_output_result(&result, 0, "query1", qlen, DIAMOND_OUTPUT_TEXT, 0);
    
    printf("\nTABULAR format header:\n");
    diamond_output_header(DIAMOND_OUTPUT_TABULAR, 0);
    printf("TABULAR format result:\n");
    diamond_output_result(&result, 0, "query1", qlen, DIAMOND_OUTPUT_TABULAR, 0);
    
    printf("\nSAM format header:\n");
    diamond_output_header(DIAMOND_OUTPUT_SAM, 0);
    printf("SAM format result:\n");
    diamond_output_result(&result, 0, "query1", qlen, DIAMOND_OUTPUT_SAM, 0);
    
    printf("\nPAF format result:\n");
    diamond_output_result(&result, 0, "query1", qlen, DIAMOND_OUTPUT_PAF, 0);
    
    /* ===== Test 6: Low-Complexity Masking ===== */
    printf("\n--- Test 6: Low-Complexity Masking (SEG) ---\n");
    
    /* Sequence with low-complexity region */
    const char* low_complex = "AAAAAAAAAAMKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAV";
    int lc_len = strlen(low_complex);
    unsigned char* lc_enc = diamond_encode_protein(low_complex, lc_len);
    
    printf("Before masking: ");
    for (int i = 0; i < 20; i++) printf("%d ", lc_enc[i]);
    printf("...\n");
    
    diamond_mask_seg(lc_enc, lc_len, 12, 220);
    
    printf("After masking:  ");
    for (int i = 0; i < 20; i++) printf("%d ", lc_enc[i]);
    printf("...\n");
    printf("(27 = masked position)\n");
    
    /* ===== Test 7: HSP Culling ===== */
    printf("\n--- Test 7: HSP Culling ---\n");
    
    DiamondResult cull_test;
    memset(&cull_test, 0, sizeof(cull_test));
    cull_test.num_hsps = 5;
    
    /* Create 5 HSPs with varying scores */
    int scores[] = {100, 200, 50, 150, 75};
    for (int i = 0; i < 5; i++) {
        cull_test.hsps[i].gapped_score = scores[i];
        cull_test.hsps[i].ungapped_score = scores[i];
    }
    
    printf("Before culling (max 3): scores=");
    for (int i = 0; i < cull_test.num_hsps; i++) {
        printf("%d ", cull_test.hsps[i].gapped_score);
    }
    printf("\n");
    
    diamond_cull_hsps(&cull_test, 3);
    
    printf("After culling: num_hsps=%d, scores=", cull_test.num_hsps);
    for (int i = 0; i < cull_test.num_hsps; i++) {
        printf("%d ", cull_test.hsps[i].gapped_score);
    }
    printf("(should be 200, 150, 100)\n");
    
    /* Cleanup */
    free(query);
    free(db);
    free(lc_enc);
    
    printf("\n=== All tests completed ===\n");
    return 0;
}
