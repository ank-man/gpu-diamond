# Makefile for GPU-Diamond
# CUDA-accelerated seed-and-extend protein alignment

CC      := gcc
CXX     := g++
NVCC    ?= nvcc

# Paths
SRC_DIR  := .
CUDA_LIB ?= /usr/local/cuda/lib64
CUDA_INC ?= /usr/local/cuda/include

CFLAGS    := -O2 -Wall -I$(SRC_DIR)/basic -I$(SRC_DIR)/search -I$(SRC_DIR)/align -I$(SRC_DIR)/util
NVCCFLAGS := -O3 --compiler-options -fno-strict-aliasing -DUNIX -I$(SRC_DIR)/basic

# Source directories
BASIC_DIR := basic
SEARCH_DIR := search
TEST_DIR := test

# Targets
all: test_gpu_diamond

# Object files
$(BASIC_DIR)/gpu_diamond.o: $(BASIC_DIR)/gpu_diamond.c $(BASIC_DIR)/gpu_diamond.h
	$(CC) $(CFLAGS) -c $< -o $@

$(SEARCH_DIR)/gpu_diamond.cu.o: $(SEARCH_DIR)/gpu_diamond.cu $(BASIC_DIR)/gpu_diamond.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Static library
libgpudiamond.a: $(BASIC_DIR)/gpu_diamond.o $(SEARCH_DIR)/gpu_diamond.cu.o
	ar rcs $@ $^

# Test executable
test_gpu_diamond: $(TEST_DIR)/test_gpu_diamond.c libgpudiamond.a
	$(CC) $(CFLAGS) -c $< -o $(TEST_DIR)/test_gpu_diamond.o
	$(CXX) -o $@ $(TEST_DIR)/test_gpu_diamond.o -L. -lgpudiamond \
	       -L$(CUDA_LIB) -lcudart

clean:
	rm -f $(BASIC_DIR)/*.o $(SEARCH_DIR)/*.cu.o $(TEST_DIR)/*.o
	rm -f *.a test_gpu_diamond

.PHONY: all clean
