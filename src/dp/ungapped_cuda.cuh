/****
GPU-Diamond CUDA ungapped extension device functions
Like real Diamond's dp/ungapped.h but for CUDA device code
****/

#ifndef UNGAPPED_CUDA_CUH
#define UNGAPPED_CUDA_CUH

#include <cuda_runtime.h>

/* CUDA device function for X-drop ungapped extension
 * Called from GPU kernel threads
 */
__device__ inline int cuda_ungapped_extend_left(
    const unsigned char* query, int q_len, int q_anchor,
    const unsigned char* subject, int s_len, int s_anchor,
    int xdrop, const signed char* score_matrix,
    int* out_q_start, int* out_s_start)
{
    int score = 0, st = 0, delta = 0, n = 1;
    int q = q_anchor - 1;
    int s = s_anchor - 1;
    
    while (score - st < xdrop && q >= 0 && s >= 0) {
        st += score_matrix[query[q] * 28 + subject[s]];
        if (st > score) {
            score = st;
            delta = n;
        }
        q--; s--; n++;
    }
    
    *out_q_start = q_anchor - delta;
    *out_s_start = s_anchor - delta;
    return score;
}

__device__ inline int cuda_ungapped_extend_right(
    const unsigned char* query, int q_len, int q_anchor,
    const unsigned char* subject, int s_len, int s_anchor,
    int xdrop, const signed char* score_matrix,
    int* out_q_end, int* out_s_end)
{
    int score = 0, st = 0, len = 0, n = 1;
    int q = q_anchor;
    int s = s_anchor;
    
    while (score - st < xdrop && q < q_len && s < s_len) {
        st += score_matrix[query[q] * 28 + subject[s]];
        if (st > score) {
            score = st;
            len = n;
        }
        q++; s++; n++;
    }
    
    *out_q_end = q_anchor + len - 1;
    *out_s_end = s_anchor + len - 1;
    return score;
}

/* Full ungapped extension on GPU */
__device__ inline int cuda_ungapped_extend_full(
    const unsigned char* query, int q_len,
    const unsigned char* subject, int s_len,
    int q_anchor, int s_anchor,
    int xdrop, const signed char* score_matrix,
    int* q_start, int* q_end,
    int* s_start, int* s_end)
{
    int left_score = cuda_ungapped_extend_left(
        query, q_len, q_anchor,
        subject, s_len, s_anchor,
        xdrop, score_matrix, q_start, s_start);
    
    int right_score = cuda_ungapped_extend_right(
        query, q_len, q_anchor,
        subject, s_len, s_anchor,
        xdrop, score_matrix, q_end, s_end);
    
    int seed_score = score_matrix[query[q_anchor] * 28 + subject[s_anchor]];
    
    return left_score + seed_score + right_score;
}

#endif
