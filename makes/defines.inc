# main paths
ROOT_DIR=$(shell dirname $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST)))))
OUTDIR=$(ROOT_DIR)/build

GTEST_LIB_PATH=/usr/lib

COMMON_CFLAGS=-std=c++14 -Wall -I/$(ROOT_DIR)/includes -DCUDA_AVAILABLE
COMMON_LFLAGS=

CC=g++
NVCC=nvcc

# clang-format
CLANG_FORMAT=clang-format
CLANG_FORMAT_ARGS=-style=file -i
