include ../../../makes/defines.inc

OUTDIR_R=$(OUTDIR)/lib_objs_r
OUTDIR_D=$(OUTDIR)/lib_objs_d

CU_SOURCES=$(wildcard *.cu)
OBJS_R=$(patsubst %.cu,$(OUTDIR_R)/%.cu.o,$(CU_SOURCES))
OBJS_D=$(patsubst %.cu,$(OUTDIR_D)/%.cu.o,$(CU_SOURCES))

COMMON_CFLAGS = -arch=sm_30 -std=c++14 -DCUDA_AVAILABLE --compiler-options '-fPIC' -I$(ROOT_DIR)/includes -I$(ROOT_DIR)/core
CFLAGS_R=$(COMMON_CFLAGS) -O3
CFLAGS_D=$(COMMON_CFLAGS) -g

.PHONY: release debug format

all: release debug format

release: $(OBJS_R)
debug: $(OBJS_D)

#### COMPILING

$(OUTDIR_R)/%.o: % | $(OUTDIR_R)
	@echo "Compiling release: $<"
	@$(NVCC) -c -o $@ $(CFLAGS_R) $<

$(OUTDIR_D)/%.o: % | $(OUTDIR_D)
	@echo "Compiling debug: $<"
	@$(NVCC) -c -o $@ $(CFLAGS_D) $<

#### DIRECTORIES

$(OUTDIR_R):
	@mkdir -p $@

$(OUTDIR_D):
	@mkdir -p $@

#### clang-format
format:
	$(CLANG_FORMAT) $(CLANG_FORMAT_ARGS) $(CU_SOURCES)
