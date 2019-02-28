include ./makes/defines.inc

.PHONY: release debug clean ctags format help

SAMPLE_NAMES = toySample mnist mnist_conv cifar10_conv

all: release debug

release:
	@+make -C core release
	@+make -C tests release
	$(foreach sample,$(SAMPLE_NAMES),make -C samples/$(sample) release;)

debug:
	@+make -C core debug
	@+make -C tests debug
	$(foreach sample,$(SAMPLE_NAMES),make -C samples/$(sample) debug;)

# utils

clean:
	rm -r $(OUTDIR)

ctags:
	ctags -R --tag-relative=yes --exclude=.git $(ROOT_DIR)

format:
	$(CLANG_FORMAT) $(CLANG_FORMAT_ARGS) $(ROOT_DIR)/includes/*.h
	@+make -C core format
	@+make -C tests format
	@+make -C samples/toySample format
	@+make -C samples/mnist format
	@+make -C samples/mnist_conv format
	@+make -C samples/cifar10_conv format

tidy:
	@+make -C core tidy
	@+make -C samples/toySample tidy
	@+make -C samples/mnist tidy
	@+make -C samples/mnist_conv tidy
	@+make -C samples/cifar10_conv tidy

help:
	@echo "Possible commands:"
	@echo "\tall     - release + debug"
	@echo "\trelease - release version of library"
	@echo "\tdebug   - debug version of library"
	@echo "\tclean   - cleans all files"
	@echo "\tctags   - creates tags for all files"
	@echo "\tformat  - runs clang-format for all source files"
