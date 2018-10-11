include ../makes/defines.inc

OUTDIR_R=$(OUTDIR)/lib_objs_r
OUTDIR_D=$(OUTDIR)/lib_objs_d

SHARED_R=$(OUTDIR)/libdll.so
SHARED_D=$(OUTDIR)/libdll_debug.so
STATIC_R=$(OUTDIR)/libdll_static.a
STATIC_D=$(OUTDIR)/libdll_static_debug.a

CPP_SOURCES=graph.cpp tensor.cpp
OBJS_R=$(patsubst %.cpp,$(OUTDIR_R)/%.o,$(CPP_SOURCES))
OBJS_D=$(patsubst %.cpp,$(OUTDIR_D)/%.o,$(CPP_SOURCES))

CFLAGS_R=$(COMMON_CFLAGS) -O3 -fPIC
CFLAGS_D=$(COMMON_CFLAGS) -g -fPIC

all: release debug

release: $(SHARED_R) $(STATIC_R)
debug: $(SHARED_D) $(STATIC_D)

#### LINKING

$(SHARED_R): $(OBJS_R)
	@echo "Linking: $@"
	@$(CC) -o $@ $^ $(LFLAGS_R) -fPIC -shared

$(SHARED_D): $(OBJS_D)
	@echo "Linking: $@" -fPIC -shared
	@$(CC) -o $@ $^ $(LFLAGS_D) -fPIC -shared

$(STATIC_R): $(OBJS_R)
	@echo "Linking: $@"
	@ar rs $@ $^

$(STATIC_D): $(OBJS_D)
	@echo "Linking: $@"
	@ar rs $@ $^

#### COMPILING

$(OUTDIR_R)/%.o: %.cpp | $(OUTDIR_R)
	@echo "Compiling release: $^"
	@$(CC) -c -o $@ $(CFLAGS_R) $^

$(OUTDIR_D)/%.o: %.cpp | $(OUTDIR_D)
	@echo "Compiling debug: $^"
	@$(CC) -c -o $@ $(CFLAGS_D) $^

#### DIRECTORIES

$(OUTDIR_R):
	@mkdir -p $@

$(OUTDIR_D):
	@mkdir -p $@
