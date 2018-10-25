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

COMMON_CFLAGS += -std=c++17
CFLAGS_R=$(COMMON_CFLAGS) -O3 -fPIC
CFLAGS_D=$(COMMON_CFLAGS) -g -fPIC

EXPORTS_MAP=$(ROOT_DIR)/core/exports.map
COMMON_LFLAGS +=-fPIC -shared -Wl,--version-script=$(EXPORTS_MAP)
LFLAGS_R=$(COMMON_LFLAGS)
LFLAGS_D=$(COMMON_LFLAGS)

.PHONY: release debug format

all: release debug

release: $(SHARED_R) $(STATIC_R)
debug: $(SHARED_D) $(STATIC_D)

#### LINKING

$(SHARED_R): $(OBJS_R)
	@echo "Linking: $@"
	@$(CC) -o $@ $^ $(LFLAGS_R)

$(SHARED_D): $(OBJS_D)
	@echo "Linking: $@"
	@$(CC) -o $@ $^ $(LFLAGS_D)

$(STATIC_R): $(OBJS_R)
	@echo "Linking: $@"
	@ar rs $@ $^

$(STATIC_D): $(OBJS_D)
	@echo "Linking: $@"
	@ar rs $@ $^

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
