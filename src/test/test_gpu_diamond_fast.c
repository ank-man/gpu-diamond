/****
GPU-Diamond UltraFast Test
Compares performance against original version
****/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../basic/gpu_diamond.h"
#include "../search/gpu_diamond_fast.h"

/* Sample protein sequences */
static const char* query_protein = 
    "MKTAYIAIVGATGAVGRIIELQGGLQPEAAESVVVASARSAGTTLQYGDDAVAAVDAVPEAAGDVVIDFTS"
    "AFGNQAVPVLSGAAGAVGAGFAAGLSSVGVVSADAGAVYDAAVEYGAQHDADLVVTMLTGGGAYTVRLLG"
    "EAGVAVAGSAAGTIGAQLAGLVAGAVAALRGAGAVAQVDAVAAGLADAA";

static const char* subjects[] = {
    /* Perfect match */
    "MKTAYIAIVGATGAVGRIIELQGGLQPEAAESVVVASARSAGTTLQYGDDAVAAVDAVPEAAGDVVIDFTS"
    "AFGNQAVPVLSGAAGAVGAGFAAGLSSVGVVSADAGAVYDAAVEYGAQHDADLVVTMLTGGGAYTVRLLG"
    "EAGVAVAGSAAGTIGAQLAGLVAGAVAALRGAGAVAQVDAVAAGLADAA",
    
    /* Partial match (60% similar) */
    "MKTAYIAIVGATGAVGRIIELQGGLQPEAAESVVVASARSAGTTLQYGDDAVAAVDAVPEAAGDVVIDFTS"
    "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    "EAGVAVAGSAAGTIGAQLAGLVAGAVAALRGAGAVAQVDAVAAGLADAA",
    
    /* Unrelated */
    "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLM"
    "NPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYAC",
    
    /* Too short (< 4 aa) */
    "MK",
};

static double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int main() {
    printf("GPU-Diamond UltraFast Performance Test\n");
    printf("========================================\n\n");
    
    int qlen = strlen(query_protein);
    int num_subjects = 4;
    
    /* Encode query */
    unsigned char* query = diamond_encode_protein(query_protein, qlen);
    printf("Query: %d aa\n\n", qlen);
    
    /* Prepare database */
    int seq_lens[4];
    int max_len = 0;
    for (int i = 0; i < num_subjects; i++) {
        seq_lens[i] = strlen(subjects[i]);
        if (seq_lens[i] > max_len) max_len = seq_lens[i];
    }
    int padded = max_len + 4;
    
    unsigned char* db = calloc((size_t)num_subjects * padded, 1);
    for (int i = 0; i < num_subjects; i++) {
        unsigned char* enc = diamond_encode_protein(subjects[i], seq_lens[i]);
        memcpy(db + (size_t)i * padded, enc, seq_lens[i]);
        free(enc);
    }
    printf("Database: %d seqs, max=%d, padded=%d\n\n", num_subjects, max_len, padded);
    
    /* === Test 1: Original API (allocates every time) === */
    printf("--- Test 1: Original API (reallocates per query) ---\n");
    
    DiamondGPUOptions opts = {
        .num_blocks = 32, .num_threads = 64,
        .num_sequences = num_subjects, .padded_length = padded,
        .x_drop = 16, .min_score = 25, .seed_score_min = 12,
    };
    
    DiamondResult* results_orig = calloc(num_subjects, sizeof(DiamondResult));
    
    double t1 = get_time_ms();
    int num_runs = 10;
    for (int run = 0; run < num_runs; run++) {
        diamond_gpu_search(query, qlen, db, seq_lens, &opts, results_orig);
    }
    double t2 = get_time_ms();
    double time_orig = (t2 - t1) / num_runs;
    
    printf("Average time: %.3f ms/query (over %d runs)\n", time_orig, num_runs);
    printf("Throughput: %.1f queries/sec\n\n", 1000.0 / time_orig);
    
    /* === Test 2: UltraFast API (persistent DB) === */
    printf("--- Test 2: UltraFast API (persistent DB) ---\n");
    
    /* Initialize persistent database once */
    GPUDiamondDB fast_db;
    int ret = gpudiamond_db_init(&fast_db, db, seq_lens, num_subjects, padded);
    if (ret != 0) {
        fprintf(stderr, "Failed to initialize fast DB: %d\n", ret);
        return 1;
    }
    
    DiamondResult* results_fast = calloc(num_subjects, sizeof(DiamondResult));
    
    /* Warmup */
    gpudiamond_search_fast(&fast_db, query, qlen, &opts, results_fast);
    
    /* Benchmark */
    t1 = get_time_ms();
    for (int run = 0; run < num_runs; run++) {
        gpudiamond_search_fast(&fast_db, query, qlen, &opts, results_fast);
    }
    t2 = get_time_ms();
    double time_fast = (t2 - t1) / num_runs;
    
    printf("Average time: %.3f ms/query (over %d runs)\n", time_fast, num_runs);
    printf("Throughput: %.1f queries/sec\n", 1000.0 / time_fast);
    printf("Speedup: %.2fx faster than original\n\n", time_orig / time_fast);
    
    /* Print stats */
    gpudiamond_print_stats(&fast_db);
    
    /* === Test 3: Multiple different queries on same DB === */
    printf("--- Test 3: Multiple queries on persistent DB ---\n");
    
    const char* queries[] = {
        "MKTAYIAIVGATGAVGRIIELQGGLQPEAAESVVVASARSAGTTLQYGDDAVAAVDAVPEAA",
        "EAGVAVAGSAAGTIGAQLAGLVAGAVAALRGAGAVAQVDAVAAGLADAA",
        "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIK",
    };
    int num_queries = 3;
    
    t1 = get_time_ms();
    for (int i = 0; i < num_queries; i++) {
        unsigned char* q = diamond_encode_protein(queries[i], strlen(queries[i]));
        gpudiamond_search_fast(&fast_db, q, strlen(queries[i]), &opts, results_fast);
        free(q);
    }
    t2 = get_time_ms();
    printf("Time for %d different queries: %.3f ms (%.3f ms avg)\n\n",
           num_queries, t2 - t1, (t2 - t1) / num_queries);
    
    /* Cleanup */
    gpudiamond_db_destroy(&fast_db);
    
    /* Verify results match */
    printf("--- Verification ---\n");
    int matches = 0;
    for (int i = 0; i < num_subjects; i++) {
        if (results_orig[i].num_hsps == results_fast[i].num_hsps) {
            matches++;
        }
    }
    printf("Results match: %d/%d sequences\n", matches, num_subjects);
    
    /* Print sample results */
    printf("\nSample results (Fast API):\n");
    for (int i = 0; i < num_subjects; i++) {
        printf("[%d] hsps=%d total_score=%d seed_hits=%d\n",
               i, results_fast[i].num_hsps, results_fast[i].total_score,
               results_fast[i].seed_hits);
        for (int j = 0; j < results_fast[i].num_hsps && j < 2; j++) {
            DiamondHSP* h = &results_fast[i].hsps[j];
            printf("    HSP%d: score=%d Q[%d-%d] S[%d-%d]\n",
                   j, h->score, h->q_start, h->q_end, h->s_start, h->s_end);
        }
    }
    
    free(query);
    free(db);
    free(results_orig);
    free(results_fast);
    
    printf("\n=== All tests completed ===\n");
    return 0;
}
