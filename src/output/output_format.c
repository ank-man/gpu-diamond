/****
GPU-Diamond output formatters implementation
****/
#include "output_format.h"
#include <stdio.h>

void output_header(OutputFormat fmt, int fields)
{
    (void)fields;
    
    switch (fmt) {
        case OUTPUT_TABULAR:
            printf("# Fields: query id, subject id, %% identity, alignment length, ");
            printf("mismatches, gap opens, q. start, q. end, s. start, s. end, ");
            printf("evalue, bit score\n");
            break;
        case OUTPUT_SAM:
            printf("@HD\tVN:1.0\tSO:unsorted\n");
            break;
        default:
            break;
    }
}

void output_hsp(const DiamondHSP* hsp, int hsp_num,
                const char* qname, int qlen,
                const char* sname, int slen,
                OutputFormat fmt, int fields)
{
    (void)fields;
    (void)hsp_num;
    (void)qlen;
    (void)slen;
    
    switch (fmt) {
        case OUTPUT_TEXT:
            printf("HSP %d: score=%d, Q[%d-%d] S[%d-%d]\n",
                   hsp_num, hsp->score,
                   hsp->q_start + 1, hsp->q_end + 1,
                   hsp->s_start + 1, hsp->s_end + 1);
            break;
            
        case OUTPUT_TABULAR:
            printf("%s\t%s\t%.1f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.2e\t%d\n",
                   qname, sname,
                   0.0,  /* pident - not computed yet */
                   hsp->q_end - hsp->q_start + 1,
                   0,  /* mismatches */
                   0,  /* gap opens */
                   hsp->q_start + 1,
                   hsp->q_end + 1,
                   hsp->s_start + 1,
                   hsp->s_end + 1,
                   0.0,  /* evalue */
                   hsp->score);
            break;
            
        default:
            break;
    }
}

void output_footer(OutputFormat fmt)
{
    (void)fmt;
    /* No footer needed for most formats */
}
