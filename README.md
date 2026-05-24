# GPU-Diamond

CUDA-accelerated seed-and-extend protein alignment.

A self-contained GPU implementation inspired by [Diamond](https://github.com/bbuchfink/diamond) 
and the GPU-BLAST architecture. Uses a 4-mer seed-and-extend algorithm with 
BLOSUM62 scoring and X-dropoff ungapped extension.

## Features

- **Pure CUDA**: No dependencies on NCBI-BLAST source tree
- **Self-contained**: Single header + 2 source files
- **Fast lookup**: Polynomial hash table for 4-mer seeds (28⁴ = 614K entries)
- **Diagonal deduplication**: 256-entry per-thread table avoids redundant extensions
- **Portable**: Works on any CUDA-capable GPU (Compute Capability 2.0+)

## Directory Structure (like real Diamond)

```
gpu-diamond/
├── CMakeLists.txt          # CMake build configuration
├── Makefile               # Alternative Make build
├── README.md              # This file
├── src/
│   ├── basic/             # Core types and constants
│   │   ├── gpu_diamond.h   # Public API header
│   │   └── gpu_diamond.c   # Host helper functions
│   ├── search/            # Seed finding and hit detection
│   │   └── gpu_diamond.cu  # CUDA kernel implementation
│   ├── data/              # Sequence and block data structures
│   │   ├── sequence.h
│   │   └── sequence.c
│   ├── stats/             # Karlin-Altschul statistics
│   │   ├── karlin_altschul.h
│   │   └── karlin_altschul.c
│   ├── masking/           # Low-complexity masking
│   │   ├── seg.h
│   │   └── seg.c
│   ├── dp/                # Dynamic programming (ungapped extension)
│   │   ├── ungapped.h
│   │   └── ungapped.c
│   ├── output/            # Output formatters
│   │   ├── output_format.h
│   │   └── output_format.c
│   ├── run/               # Main execution pipeline
│   │   ├── run.h
│   │   └── run.c
│   ├── align/             # Gapped alignment (future expansion)
│   ├── util/              # Utility functions (future expansion)
│   └── test/              # Test suite
│       └── test_gpu_diamond.c
└── .gitignore
```

| Directory | Description |
|-----------|-------------|
| `src/basic/` | Core API, types, constants, encoding functions |
| `src/search/` | CUDA kernels for seed lookup and extension |
| `src/data/` | Sequence and block data structures |
| `src/stats/` | Karlin-Altschul statistics (E-values, bit-scores) |
| `src/masking/` | SEG-style low-complexity masking |
| `src/dp/` | Dynamic programming (ungapped extension) |
| `src/output/` | Output formatters (tabular, SAM, etc.) |
| `src/run/` | Main execution pipeline |
| `src/align/` | Gapped alignment (future expansion) |
| `src/util/` | Utility functions (future expansion) |
| `src/test/` | Test suite |

## Building

### Using CMake (like real Diamond)

```bash
mkdir build
cd build
cmake ..
make
```

### Using Make

```bash
make                    # Build library and test
make clean              # Clean build artifacts
```

## Running Tests

```bash
./test_gpu_diamond
```

## Algorithm

1. **Lookup table** (host): every 4-mer in the query is hashed:
   ```
   h = a0·28³ + a1·28² + a2·28 + a3
   ```
   stored in `bucket_pos[h*32 + i]`.

2. **Kernel** (device): each thread scans every 4-mer of one (or more)
   subject sequences, looks up matching query positions, performs ungapped
   X-dropoff extension, and stores up to 32 HSPs per subject.

3. **Diagonal deduplication**: 256-entry per-thread table avoids
   redundant extensions on the same diagonal.

## Building

### Requirements
- CUDA Toolkit 7.5+ (nvcc)
- GCC/G++ 4.8+
- CUDA-capable GPU

### Build
```bash
make
```

Override CUDA paths if needed:
```bash
make NVCC=/usr/local/cuda/bin/nvcc CUDA_LIB=/usr/local/cuda/lib64
```

## Running the test

```bash
./test_gpu_diamond
```

Expected output (scores depend on BLOSUM62):

```
GPU-Diamond standalone test
===========================

Query length: 122 aa
Database: 4 sequences, max=122 aa, padded=124

Elapsed: X.XXX ms

[0] perfect_match    seed_hits=...  ext=...  hsps=1  total_score=...
     hsp0: q[0..121] s[0..121] score=...
[1] partial_match    ...
[2] unrelated        seed_hits=... hsps=0
[3] too_short        seed_hits=0 hsps=0

TEST PASSED
```

The test asserts:
- `perfect_match` produces an HSP whose score ≥ query length.
- `too_short` produces no HSPs.

## Public API

```c
#include "gpu_diamond.h"

unsigned char* q = diamond_encode_protein("MKTAY...", qlen);

DiamondGPUOptions opts = {
    .num_blocks = 32, .num_threads = 64,
    .num_sequences = N, .padded_length = padded,
    .x_drop = 16, .min_score = 25, .seed_score_min = 12,
};

DiamondResult* res = calloc(N, sizeof(DiamondResult));
diamond_gpu_search(q, qlen, db_padded, seq_lens, &opts, res);
```

Database layout: 
- `db_padded[seq_idx * padded_length + offset]`
- Encoded amino acids (0..27)
- Sequences shorter than `padded_length` are zero-padded
- Kernel uses `seq_lens[seq]` as the true length

## Limits

| Symbol | Value | Meaning |
| --- | --- | --- |
| `DIAMOND_MAX_QUERY_LEN` | 8192 | Query fits in constant memory |
| `DIAMOND_SEED_SIZE` | 4 | Contiguous 4-mer seed |
| `DIAMOND_HASH_SIZE` | 614656 | 28⁴ |
| `DIAMOND_MAX_HITS_BUCKET` | 32 | Max query positions per seed |
| `DIAMOND_MAX_HSPS` | 32 | Max HSPs returned per subject |
| `DIAMOND_DIAG_SIZE` | 256 | Per-thread diagonal dedup table |

## Known Limitations

- **Spaced seeds**: Not implemented (real Diamond uses multiple seed shapes)
- **Gapped extension**: Ungapped only; no banded Smith-Waterman
- **Statistics**: No E-value or bit-score computation
- **Batched queries**: One query at a time

This is a working baseline for GPU-accelerated seed-and-extend alignment.

## License

GPU-Diamond is based on concepts from GPU-BLAST and Diamond. 
Use and modify as needed for your research.

## References

1. Buchfink et al. (2015) "Fast and sensitive protein alignment using Diamond", Nature Methods
2. Vouzis & Sahinidis (2011) "GPU-BLAST: Using graphics processors to accelerate protein sequence alignment", Bioinformatics
