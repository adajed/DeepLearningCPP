include ../makes/defines.inc

OUTDIR_R=$(OUTDIR)/lib_objs_r
OUTDIR_D=$(OUTDIR)/lib_objs_d

SHARED_R=$(OUTDIR)/libdll.so
SHARED_D=$(OUTDIR)/libdll_debug.so
STATIC_R=$(OUTDIR)/libdll_static.a
STATIC_D=$(OUTDIR)/libdll_static_debug.a

CPP_SOURCES=$(wildcard *.cpp)
CPP_HEADERS=$(wildcard *.h)
OBJS_R=$(patsubst %.cpp,$(OUTDIR_R)/%.o,$(CPP_SOURCES))
OBJS_D=$(patsubst %.cpp,$(OUTDIR_D)/%.o,$(CPP_SOURCES))

COMMON_CFLAGS += -Wunused-parameter -I$(ROOT_DIR)/core -I$(ROOT_DIR)/core/layers -fPIC
CFLAGS_R=$(COMMON_CFLAGS) -O3
CFLAGS_D=$(COMMON_CFLAGS) -g

EXPORTS_MAP=$(ROOT_DIR)/core/exports.map
COMMON_LFLAGS +=-fPIC -shared -Wl,--version-script=$(EXPORTS_MAP)
LFLAGS_R=$(COMMON_LFLAGS)
LFLAGS_D=$(COMMON_LFLAGS)

.PHONY: release debug format layers_release layers_debug

all: release debug

release: layers_release $(SHARED_R) $(STATIC_R)
debug: layers_debug $(SHARED_D) $(STATIC_D)

#### LAYERS
layers_release:
	@+make -C layers release

layers_debug:
	@+make -C layers debug

#### LINKING

$(SHARED_R): $(OUTDIR_R) $(OBJS_R)
	@echo "Linking: $@"
	@$(CC) -o $@ $(wildcard $(OUTDIR_R)/*.o) $(LFLAGS_R)

$(SHARED_D): $(OUTDIR_D) $(OBJS_D)
	@echo "Linking: $@"
	@$(CC) -o $@ $(wildcard $(OUTDIR_D)/*.o) $(LFLAGS_D)

$(STATIC_R): $(OUTDIR_R) $(OBJS_R)
	@echo "Linking: $@"
	@ar rs $@ $(wildcard $(OUTDIR_R)/*.o)

$(STATIC_D): $(OUTDIR_D) $(OBJS_D)
	@echo "Linking: $@"
	@ar rs $@ $(wildcard $(OUTDIR_D)/*.o)

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
	@+make -C layers format

#### clang-tidy
tidy:
	@$(foreach src,$(CPP_SOURCES),\
		$(CLANG_TIDY_EXEC) --fix $(src) -quiet -- $(CLANG_TIDY_INCLUDES) $(CLANG_TIDY_ARGS);)
	@+make -C layers tidy
