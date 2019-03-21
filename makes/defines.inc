# main paths
ROOT_DIR=$(shell dirname $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST)))))
OUTDIR=$(ROOT_DIR)/build

GTEST_LIB_PATH=/usr/lib

# check for nvcc and cuda
NVCC_RESULT=$(shell which /usr/local/cuda/bin/nvcc 2> /dev/null)
NVCC_TEST=$(notdir $(NVCC_RESULT))
ifeq ($(NVCC_TEST),nvcc)
    CUDA_AVAILABLE=1
else
    CUDA_AVAILABLE=0
endif

COMMON_CFLAGS=-std=c++14 -Wall -I$(ROOT_DIR)/includes
COMMON_LFLAGS=

ifeq ($(CUDA_AVAILABLE),1)
    COMMON_CFLAGS += -DCUDA_AVAILABLE -I/usr/local/cuda/include
endif

CC=g++
NVCC=/usr/local/cuda/bin/nvcc

# clang-format
CLANG_FORMAT=clang-format
CLANG_FORMAT_ARGS=-style=file -i

# clang-tidy
CLANG_TIDY_EXEC=clang-tidy
CLANG_TIDY_INCLUDES=$(foreach dir,includes core core/layers,-I$(ROOT_DIR)/$(dir))
CLANG_TIDY_ARGS=-std=c++14
