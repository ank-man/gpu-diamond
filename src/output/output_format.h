/****
GPU-Diamond output formatters
Like real Diamond's output/output_format.h
****/

#ifndef OUTPUT_FORMAT_H
#define OUTPUT_FORMAT_H

#include "../basic/gpu_diamond.h"

typedef enum {
    OUTPUT_TEXT = 0,        /* Human readable */
    OUTPUT_TABULAR,         /* BLAST tabular format */
    OUTPUT_PAIRWISE,        /* BLAST pairwise */
    OUTPUT_SAM,             /* SAM format */
    OUTPUT_PAF              /* PAF format */
} OutputFormat;

/* Output fields bitmasks */
#define FIELD_QSEQID    0x0001
#define FIELD_SSEQID    0x0002
#define FIELD_PIDENT    0x0004
#define FIELD_LENGTH    0x0008
#define FIELD_MISMATCH  0x0010
#define FIELD_GAPOPEN   0x0020
#define FIELD_QSTART    0x0040
#define FIELD_QEND      0x0080
#define FIELD_SSTART    0x0100
#define FIELD_SEND      0x0200
#define FIELD_EVALUE    0x0400
#define FIELD_BITSCORE  0x0800
#define FIELD_SCORE     0x1000

/* Output header for format */
void output_header(OutputFormat fmt, int fields);

/* Output a single HSP */
void output_hsp(const DiamondHSP* hsp, int hsp_num,
                const char* qname, int qlen,
                const char* sname, int slen,
                OutputFormat fmt, int fields);

/* Output footer */
void output_footer(OutputFormat fmt);

#endif
