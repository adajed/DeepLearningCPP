include ../../makes/defines.inc

OUTDIR_R=$(OUTDIR)/lib_objs_r
OUTDIR_D=$(OUTDIR)/lib_objs_d

CPP_SOURCES=$(wildcard *.cpp)
CPP_HEADERS=$(wildcard *.h)
OBJS_R=$(patsubst %.cpp,$(OUTDIR_R)/%.o,$(CPP_SOURCES))
OBJS_D=$(patsubst %.cpp,$(OUTDIR_D)/%.o,$(CPP_SOURCES))

TIDY_TARGETS=$(patsubst %.cpp, .tidy_%,$(CPP_SOURCES))

COMMON_CFLAGS += -Wunused-parameter -I$(ROOT_DIR)/core -fPIC
CFLAGS_R=$(COMMON_CFLAGS) -O3
CFLAGS_D=$(COMMON_CFLAGS) -g

.PHONY: release debug format

all: release debug

release: $(OBJS_R)
debug: $(OBJS_D)

#### COMPILING

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

#### clang-tidy
tidy: $(TIDY_TARGETS)

.tidy_%: %.cpp $(CPP_HEADERS)
	$(CLANG_TIDY_EXEC) --fix $< -quiet -- $(CLANG_TIDY_INCLUDES) $(CLANG_TIDY_ARGS)
	@touch $@
