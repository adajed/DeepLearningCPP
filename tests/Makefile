include ../makes/defines.inc

OUTDIR_R=$(OUTDIR)/tests_objs_r
OUTDIR_D=$(OUTDIR)/tests_objs_d

PROGRAM_R=$(OUTDIR)/test
PROGRAM_D=$(OUTDIR)/test_debug

CPP_SOURCES=main.cpp
OBJS_R=$(patsubst %.cpp,$(OUTDIR_R)/%.o,$(CPP_SOURCES))
OBJS_D=$(patsubst %.cpp,$(OUTDIR_D)/%.o,$(CPP_SOURCES))

CFLAGS_R=$(COMMON_CFLAGS) -O3
CFLAGS_D=$(COMMON_CFLAGS) -g

COMMON_LFLAGS +=-L$(OUTDIR) -L$(GTEST_LIB_PATH)
LFLAGS_R=$(COMMON_LFLAGS) -ldll -pthread -lgtest -lgtest_main
LFLAGS_D=$(COMMON_LFLAGS) -ldll_debug -pthread -lgtest -lgtest_main

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
