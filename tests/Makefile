include ../makes/defines.inc

OUTDIR_R=$(OUTDIR)/tests_objs_r
OUTDIR_D=$(OUTDIR)/tests_objs_d

PROGRAM_R=$(OUTDIR)/test
PROGRAM_D=$(OUTDIR)/test_debug

CPP_SOURCES=$(wildcard *.cpp)
CPP_HEADERS=$(wildcard *.h)
OBJS_R=$(patsubst %.cpp,$(OUTDIR_R)/%.o,$(CPP_SOURCES))
OBJS_D=$(patsubst %.cpp,$(OUTDIR_D)/%.o,$(CPP_SOURCES))

COMMON_CFLAGS += -std=c++17 -I$(ROOT_DIR)/core -I$(ROOT_DIR)/core/layers
CFLAGS_R=$(COMMON_CFLAGS) -O3
CFLAGS_D=$(COMMON_CFLAGS) -g

COMMON_LFLAGS +=-L$(OUTDIR) -L$(GTEST_LIB_PATH)
ifeq ($(CUDA_AVAILABLE),1)
	COMMON_LFLAGS += -L/usr/local/cuda-10.0/lib64 -lcuda -lcurand -lcudart
endif
LFLAGS_R=$(COMMON_LFLAGS) -ldll_static -pthread -lgtest
LFLAGS_D=$(COMMON_LFLAGS) -ldll_static_debug -pthread -lgtest


.PHONY: release debug format

all: release debug

release: $(PROGRAM_R)
debug: $(PROGRAM_D)

#### LINKING

$(PROGRAM_R): $(OBJS_R)
	@echo "Linking: $@"
	@$(CC) $^ -o $@ $(LFLAGS_R)

$(PROGRAM_D): $(OBJS_D)
	@echo "Linking: $@"
	@$(CC) $^ -o $@ $(LFLAGS_D)

#### COMPILING

$(OUTDIR_R)/%.o: %.cpp $(CPP_HEADERS) $(OUTDIR)/libdll_static.a | $(OUTDIR_R)
	@echo "Compiling release: $<"
	@$(CC) -c -o $@ $(CFLAGS_R) $<

$(OUTDIR_D)/%.o: %.cpp $(CPP_HEADERS) $(OUTDIR)/libdll_static_debug.a | $(OUTDIR_D)
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
