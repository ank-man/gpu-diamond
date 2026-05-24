# Makefile for GPU-Diamond v2.0 (Enhanced)
# CUDA-accelerated seed-and-extend with gapped alignment

CC      := gcc
CXX     := g++
NVCC    ?= nvcc

# CUDA paths
CUDA_LIB ?= /usr/local/cuda/lib64
CUDA_INC ?= /usr/local/cuda/include

# Flags
CFLAGS    := -O2 -Wall -std=c99
NVCCFLAGS := -O3 --compiler-options -fno-strict-aliasing -DUNIX \
             -gencode arch=compute_52,code=sm_52 \
             -gencode arch=compute_61,code=sm_61 \
             -gencode arch=compute_70,code=sm_70 \
             -gencode arch=compute_80,code=sm_80

# Default target
all: test_gpu_diamond_v2

# CPU object files
gpu_diamond_v2.o: gpu_diamond_v2.c gpu_diamond_v2.h
	$(CC) $(CFLAGS) -c gpu_diamond_v2.c -o gpu_diamond_v2.o

test_gpu_diamond_v2.o: test_gpu_diamond_v2.c gpu_diamond_v2.h
	$(CC) $(CFLAGS) -c test_gpu_diamond_v2.c -o test_gpu_diamond_v2.o

# CUDA object file
gpu_diamond_v2.cu.o: gpu_diamond_v2.cu gpu_diamond_v2.h
	$(NVCC) $(NVCCFLAGS) -c gpu_diamond_v2.cu -o gpu_diamond_v2.cu.o

# Static library
libgpudiamond_v2.a: gpu_diamond_v2.o gpu_diamond_v2.cu.o
	ar rcs libgpudiamond_v2.a gpu_diamond_v2.o gpu_diamond_v2.cu.o

# Test executable
test_gpu_diamond_v2: test_gpu_diamond_v2.o libgpudiamond_v2.a
	$(CXX) -o test_gpu_diamond_v2 test_gpu_diamond_v2.o -L. -lgpudiamond_v2 \
	       -L$(CUDA_LIB) -lcudart -lm

# Simple compile check (no CUDA linking)
test_cpu_only: test_gpu_diamond_v2.o gpu_diamond_v2.o
	$(CC) -o test_cpu_only test_gpu_diamond_v2.o gpu_diamond_v2.o -lm
	@echo "Note: test_cpu_only tests CPU functions only (no GPU)"

clean:
	rm -f *.o *.a test_gpu_diamond_v2 test_cpu_only

.PHONY: all clean test_cpu_only
