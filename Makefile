# Makefile for GPU-Diamond (root Makefile)
# CUDA-accelerated seed-and-extend protein alignment
# Structure like real Diamond: src/{basic,search,align,data,stats,masking,dp,output,run,test,util}

CC      := gcc
CXX     := g++
NVCC    ?= nvcc

# Paths
SRC_DIR  := src
CUDA_LIB ?= /usr/local/cuda/lib64
CUDA_INC ?= /usr/local/cuda/include

# Include all Diamond-like directories
INCLUDES := -I$(SRC_DIR) \
            -I$(SRC_DIR)/basic \
            -I$(SRC_DIR)/search \
            -I$(SRC_DIR)/align \
            -I$(SRC_DIR)/data \
            -I$(SRC_DIR)/stats \
            -I$(SRC_DIR)/masking \
            -I$(SRC_DIR)/dp \
            -I$(SRC_DIR)/output \
            -I$(SRC_DIR)/run \
            -I$(SRC_DIR)/util

CFLAGS    := -O2 -Wall $(INCLUDES)
NVCCFLAGS := -O3 --compiler-options -fno-strict-aliasing -DUNIX -I$(SRC_DIR)/basic

# Source directories
BASIC_DIR   := $(SRC_DIR)/basic
SEARCH_DIR  := $(SRC_DIR)/search
DATA_DIR    := $(SRC_DIR)/data
STATS_DIR   := $(SRC_DIR)/stats
MASKING_DIR := $(SRC_DIR)/masking
DP_DIR      := $(SRC_DIR)/dp
OUTPUT_DIR  := $(SRC_DIR)/output
RUN_DIR     := $(SRC_DIR)/run
TEST_DIR    := $(SRC_DIR)/test

# C source files (like real Diamond's structure)
C_SOURCES := \
    $(BASIC_DIR)/gpu_diamond.c \
    $(DATA_DIR)/sequence.c \
    $(STATS_DIR)/karlin_altschul.c \
    $(MASKING_DIR)/seg.c \
    $(DP_DIR)/ungapped.c \
    $(OUTPUT_DIR)/output_format.c \
    $(RUN_DIR)/run.c

# CUDA source files
CU_SOURCES := \
    $(SEARCH_DIR)/gpu_diamond.cu \
    $(SEARCH_DIR)/gpu_diamond_fast.cu

# Object files
C_OBJECTS := $(C_SOURCES:.c=.o)
CU_OBJECTS := $(CU_SOURCES:.cu=.cu.o)

# Targets
all: test_gpu_diamond

# Pattern rules for C files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Pattern rules for CUDA files
%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Static library (includes all modules like real Diamond)
libgpudiamond.a: $(C_OBJECTS) $(CU_OBJECTS)
	ar rcs $@ $^

# Test executables
all: test_gpu_diamond test_gpu_diamond_fast

test_gpu_diamond: $(TEST_DIR)/test_gpu_diamond.c libgpudiamond.a
	$(CC) $(CFLAGS) -c $< -o $(TEST_DIR)/test_gpu_diamond.o
	$(CXX) -o $@ $(TEST_DIR)/test_gpu_diamond.o -L. -lgpudiamond \
	       -L$(CUDA_LIB) -lcudart -lm

# UltraFast test (persistent DB, early termination)
test_gpu_diamond_fast: $(TEST_DIR)/test_gpu_diamond_fast.c libgpudiamond.a
	$(CC) $(CFLAGS) -c $< -o $(TEST_DIR)/test_gpu_diamond_fast.o
	$(CXX) -o $@ $(TEST_DIR)/test_gpu_diamond_fast.o -L. -lgpudiamond \
	       -L$(CUDA_LIB) -lcudart -lm

# CPU-only test (no CUDA linking)
test_cpu_only: $(C_OBJECTS) $(TEST_DIR)/test_gpu_diamond.c
	$(CC) $(CFLAGS) -DCPU_ONLY -c $(TEST_DIR)/test_gpu_diamond.c -o $(TEST_DIR)/test_gpu_diamond.o
	$(CC) -o test_cpu_only $(C_OBJECTS) $(TEST_DIR)/test_gpu_diamond.o -lm

clean:
	rm -f $(BASIC_DIR)/*.o $(SEARCH_DIR)/*.cu.o
	rm -f $(DATA_DIR)/*.o $(STATS_DIR)/*.o $(MASKING_DIR)/*.o
	rm -f $(DP_DIR)/*.o $(OUTPUT_DIR)/*.o $(RUN_DIR)/*.o
	rm -f $(TEST_DIR)/*.o
	rm -f *.a test_gpu_diamond test_gpu_diamond_fast test_cpu_only

.PHONY: all clean test_cpu_only

# Show structure (like Diamond)
print-structure:
	@echo "GPU-Diamond source structure (like real Diamond):"
	@echo "  src/basic/   - Core types and constants"
	@echo "  src/search/  - CUDA kernels for seed finding"
	@echo "  src/data/    - Sequence and block data structures"
	@echo "  src/stats/   - Karlin-Altschul statistics"
	@echo "  src/masking/ - SEG low-complexity masking"
	@echo "  src/dp/      - Dynamic programming (ungapped extension)"
	@echo "  src/output/  - Output formatters (tabular, SAM, etc.)"
	@echo "  src/run/     - Main execution pipeline"
	@echo "  src/test/    - Test suite"
	@echo "  src/util/    - Utility functions (future)"
	@echo "  src/align/   - Gapped alignment (future)"

