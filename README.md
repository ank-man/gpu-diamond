# GPU-Diamond v2.0

CUDA-accelerated seed-and-extend protein alignment with real Diamond features.

A self-contained GPU implementation inspired by [Diamond](https://github.com/bbuchfink/diamond) 
and the GPU-BLAST architecture. **Now with spaced seeds, gapped alignment, 
E-values, and multiple output formats!**

## Features

- **Pure CUDA**: No dependencies on NCBI-BLAST source tree
- **Self-contained**: Single header + 2 source files
- **Spaced Seeds**: Multiple seed shapes (e.g., `111101110111`)
- **Gapped Extension**: Banded Smith-Waterman on GPU
- **Statistics**: Karlin-Altschul E-values and bit-scores
- **Output Formats**: BLAST tabular, SAM, PAF, text
- **Sensitivity Modes**: `--faster`, `--fast`, `--sensitive`, `--more-sensitive`
- **Low-Complexity Masking**: SEG-style masking
- **HSP Culling**: Keep top N HSPs per target

## Files

| File | Description |
|------|-------------|
| `gpu_diamond.h` | Public API and configuration |
| `gpu_diamond.c` | Host: AA encoding, BLOSUM62, spaced seeds, statistics |
| `gpu_diamond.cu` | CUDA kernel: seed-and-extend + banded SW |
| `test_gpu_diamond.c` | Comprehensive test suite |
| `Makefile` | Build rules (no external deps) |

## New in v2.0

### 1. Spaced Seeds
Diamond uses sophisticated seed patterns like `111101110111` instead of just contiguous 4-mers:

```c
DiamondSeedShape shape;
diamond_seed_shape_init(&shape, "111101110111");  /* weight=10, length=12 */

/* Get default shapes for sensitivity level */
int count;
const char** shapes = diamond_get_default_shapes(DIAMOND_SENSITIVITY_SENSITIVE, &count);
/* Returns 16 shapes for sensitive mode */
```

### 2. Sensitivity Presets
Match Diamond's sensitivity modes:

```c
DiamondGPUOptions opts;
diamond_set_sensitivity(&opts, DIAMOND_SENSITIVITY_DEFAULT);
/* Configures xdrop, band width, gap penalties automatically */
```

| Mode | Gap Mode | X-drop | Band Width | Use Case |
|------|----------|--------|------------|----------|
| FASTER | Ungapped | 12 | 16 | Maximum speed |
| FAST | Ungapped | 16 | 16 | Fast screening |
| DEFAULT | Banded | 16/20 | 32 | Balanced |
| SENSITIVE | Banded | 16/20 | 48 | Better sensitivity |
| MORE_SENSITIVE | Banded Slow | 16/20 | 48 | High sensitivity |

### 3. Gapped Extension (Banded SW)

```c
opts.gap_mode = DIAMOND_GAP_BANDED_FAST;
opts.gap_open = 11;      /* BLOSUM62 defaults */
opts.gap_extend = 1;
opts.band_width = 32;    /* Narrow band for speed */
```

### 4. Karlin-Altschul Statistics

```c
DiamondKarlinAltschul ka;
diamond_init_karlin_altschul(&ka, 11, 1);  /* gap (11,1) */

double bitscore = diamond_bitscore(raw_score, &ka);
double evalue = diamond_evalue(raw_score, query_len, &db_stats, &ka);
```

### 5. Output Formats

```c
/* BLAST tabular */
diamond_output_header(DIAMOND_OUTPUT_TABULAR, 0);
diamond_output_result(&result, 0, "query1", qlen, DIAMOND_OUTPUT_TABULAR, 0);
/* query1\tsubject1\t85.7\t35\t5\t2\t9\t43\t4\t38\t1e-25\t45.2 */

/* SAM format */
diamond_output_header(DIAMOND_OUTPUT_SAM, 0);
diamond_output_result(&result, 0, "query1", qlen, DIAMOND_OUTPUT_SAM, 0);

/* PAF format (minimap2 compatible) */
diamond_output_result(&result, 0, "query1", qlen, DIAMOND_OUTPUT_PAF, 0);
```

### 6. Low-Complexity Masking

```c
/* Mask low-complexity regions (SEG-style) */
diamond_mask_seg(query, qlen, 12, 220);  /* window=12, trigger=2.2 entropy */
```

### 7. HSP Culling

```c
/* Keep only top 10 HSPs per target by gapped score */
opts.max_hsps_per_target = 10;
diamond_cull_hsps(&result, 10);
```

## Building

### Requirements
- CUDA Toolkit 7.5+ (nvcc)
- GCC/G++ 4.8+
- CUDA-capable GPU (Compute Capability 5.0+)

### Build
```bash
make                    # Full build with CUDA
make test_cpu_only      # CPU-only test (no GPU required)
```

Override CUDA paths:
```bash
make NVCC=/usr/local/cuda/bin/nvcc CUDA_LIB=/usr/local/cuda/lib64
```

## Running Tests

### CPU Features Test (no GPU required)
```bash
make test_cpu_only
./test_cpu_only
```

Tests:
- Contiguous 4-mer lookup tables
- Spaced seed hashing
- Karlin-Altschul statistics
- Sensitivity presets
- Output formats (tabular, SAM, PAF)
- Low-complexity masking
- HSP culling

### Full GPU Test
```bash
make
./test_gpu_diamond
```

## API Example

```c
#include "gpu_diamond.h"

/* Encode query */
unsigned char* query = diamond_encode_protein("MKTAY...", qlen);

/* Mask low-complexity regions */
diamond_mask_seg(query, qlen, 12, 220);

/* Set sensitivity mode */
DiamondGPUOptions opts = {0};
diamond_set_sensitivity(&opts, DIAMOND_SENSITIVITY_DEFAULT);

/* Use spaced seed shape */
DiamondSeedShape shape;
diamond_seed_shape_init(&shape, "111101110111");
opts.seed_shapes = &shape;
opts.num_shapes = 1;

/* Build lookup table */
uint64_t hash_space = 1;
for (int i = 0; i < shape.weight; i++) hash_space *= 28;
uint32_t *bucket_count = calloc(hash_space, sizeof(uint32_t));
uint32_t *bucket_pos = calloc(hash_space * 32, sizeof(uint32_t));
diamond_build_lookup_spaced(query, qlen, &shape, bucket_count, bucket_pos);

/* Run search */
DiamondResult* results = calloc(num_seqs, sizeof(DiamondResult));
diamond_gpu_search(query, qlen, db_padded, seq_lens, &opts, results);

/* Compute statistics */
DiamondDBStats db_stats = {.db_size = 1000000, .db_seqs = 5000};
DiamondKarlinAltschul ka;
diamond_init_karlin_altschul(&ka, 11, 1);
opts.db_stats = &db_stats;
opts.ka_params = &ka;
diamond_compute_stats(results, num_seqs, qlen, &opts);

/* Output */
diamond_output_header(DIAMOND_OUTPUT_TABULAR, 0);
for (int i = 0; i < num_seqs; i++) {
    diamond_output_result(&results[i], i, "query1", qlen, 
                        DIAMOND_OUTPUT_TABULAR, 0);
}
```

## Configuration Limits

| Symbol | Value | Meaning |
|--------|-------|---------|
| `DIAMOND_MAX_QUERY_LEN` | 8192 | Max query length |
| `DIAMOND_MAX_SEED_SHAPES` | 16 | Max seed patterns |
| `DIAMOND_MAX_SEED_WEIGHT` | 10 | Max '1's in pattern |
| `DIAMOND_MAX_SEED_LENGTH` | 32 | Max pattern length |
| `DIAMOND_MAX_HSPS` | 32 | Max HSPs per target |
| `DIAMOND_MAX_HITS_BUCKET` | 32 | Max query positions per seed |
| `MAX_BAND_WIDTH` | 64 | Max band width for SW |

## Implementation Status

| Feature | Status | Notes |
|---------|--------|-------|
| Contiguous seeds | ✅ | 4-mer default |
| Spaced seeds | ✅ | Multiple patterns |
| Ungapped extension | ✅ | X-dropoff |
| Gapped extension | ✅ | Banded SW (basic) |
| K-A statistics | ✅ | E-value, bitscore |
| Output formats | ✅ | Tabular, SAM, PAF |
| Sensitivity modes | ✅ | 5 presets |
| Masking | ✅ | SEG-style |
| HSP culling | ✅ | By score |
| Multiple queries | 🚧 | One at a time |
| Frameshift | ❌ | Future work |
| Clustering | ❌ | Not planned |

## License

GPU-Diamond is based on concepts from GPU-BLAST and Diamond.
Use and modify as needed for your research.

## References

1. Buchfink et al. (2021) "Sensitive protein alignments at tree-of-life scale using DIAMOND", *Nature Methods* 18, 366–368
2. Buchfink et al. (2015) "Fast and sensitive protein alignment using DIAMOND", *Nature Methods* 12, 59-60
3. Vouzis & Sahinidis (2011) "GPU-BLAST: Using graphics processors to accelerate protein sequence alignment", *Bioinformatics*
