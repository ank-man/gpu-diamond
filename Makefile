# Makefile for GPU-Diamond
# CUDA-accelerated seed-and-extend protein alignment

CC      := gcc
CXX     := g++
NVCC    ?= nvcc

# CUDA paths (override if needed)
CUDA_LIB ?= /usr/local/cuda/lib64
CUDA_INC ?= /usr/local/cuda/include

CFLAGS    := -O2 -Wall
NVCCFLAGS := -O3 --compiler-options -fno-strict-aliasing -DUNIX

# Targets
all: test_gpu_diamond

gpu_diamond.o: gpu_diamond.c gpu_diamond.h
	$(CC) $(CFLAGS) -c gpu_diamond.c -o gpu_diamond.o

gpu_diamond.cu.o: gpu_diamond.cu gpu_diamond.h
	$(NVCC) $(NVCCFLAGS) -c gpu_diamond.cu -o gpu_diamond.cu.o

libgpudiamond.a: gpu_diamond.o gpu_diamond.cu.o
	ar rcs libgpudiamond.a gpu_diamond.o gpu_diamond.cu.o

test_gpu_diamond.o: test_gpu_diamond.c gpu_diamond.h
	$(CC) $(CFLAGS) -c test_gpu_diamond.c -o test_gpu_diamond.o

test_gpu_diamond: test_gpu_diamond.o libgpudiamond.a
	$(CXX) -o test_gpu_diamond test_gpu_diamond.o -L. -lgpudiamond \
	       -L$(CUDA_LIB) -lcudart

clean:
	rm -f *.o *.a test_gpu_diamond

.PHONY: all clean
