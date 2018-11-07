include ../../makes/defines.inc

OUTDIR_R=$(OUTDIR)/lib_objs_r
OUTDIR_D=$(OUTDIR)/lib_objs_d

CPP_SOURCES=$(wildcard *.cpp)
CPP_HEADERS=$(wildcard *.h)
OBJS_R=$(patsubst %.cpp,$(OUTDIR_R)/%.o,$(CPP_SOURCES))
OBJS_D=$(patsubst %.cpp,$(OUTDIR_D)/%.o,$(CPP_SOURCES))

COMMON_CFLAGS += -std=c++17 -I$(ROOT_DIR)/core -fPIC
CFLAGS_R=$(COMMON_CFLAGS) -O3
CFLAGS_D=$(COMMON_CFLAGS) -g

.PHONY: release debug cuda_release cuda_debug format

all: release debug

release: cuda_release $(OBJS_R)
debug: cuda_debug $(OBJS_D)

#### COMPILING

cuda_release:
ifeq ($(CUDA_AVAILABLE),1)
	@+make -C cuda release
endif

cuda_debug:
ifeq ($(CUDA_AVAILABLE),1)
	@+make -C cuda debug
endif

$(OUTDIR_R)/%.o: %.cpp $(CPP_HEADERS) | $(OUTDIR_R)
	@echo "Compiling release: $<"
	@$(CC) -c -o $@ $(CFLAGS_R) $<

$(OUTDIR_D)/%.o: %.cpp $(CPP_HEADERS) | $(OUTDIR_D)
	@echo "Compiling debug: $<"
	@$(CC) -c -o $@ $(CFLAGS_D) $<

#### DIRECTORIES

$(OUTDIR_R):
	@mkdir -p $@

$(OUTDIR_D):
	@mkdir -p $@

#### clang-format
format:
	$(CLANG_FORMAT) $(CLANG_FORMAT_ARGS) $(CPP_SOURCES) $(CPP_HEADERS)
	@+make -C cuda format

#### clang-tidy
tidy:
	@$(foreach src,$(CPP_SOURCES),\
		$(CLANG_TIDY_EXEC) --fix $(src) -quiet -- $(CLANG_TIDY_INCLUDES) $(CLANG_TIDY_ARGS);)
