# main paths
ROOT_DIR=$(shell dirname $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST)))))
OUTDIR=$(ROOT_DIR)/build

GTEST_INCLUDE=/usr/local/include
GTEST_LIB_PATH=/usr/local/lib

# cuda settings

USE_CUDA ?= 0
CUDA_PATH ?= /usr/local/cuda

COMMON_CFLAGS=-std=c++14 -Wall -Wno-sign-compare -I$(ROOT_DIR)/includes
COMMON_LFLAGS=

ifeq ($(USE_CUDA),1)
    COMMON_CFLAGS += -DCUDA_AVAILABLE -I$(CUDA_PATH)/include
endif

# compilers
CC=g++
NVCC=$(CUDA_PATH)/bin/nvcc

# clang-format
CLANG_FORMAT=clang-format
CLANG_FORMAT_ARGS=-style=file -i

# clang-tidy
CLANG_TIDY_EXEC=clang-tidy
CLANG_TIDY_INCLUDES=$(foreach dir,includes core core/layers,-I$(ROOT_DIR)/$(dir))
CLANG_TIDY_ARGS=-std=c++14
