include ./makes/defines.inc

.PHONY: release debug clean ctags format help

all: release debug

release:
	@+make -C core release
	@+make -C tests release
	@+make -C samples/toySample release
	@+make -C samples/mnist release
	@+make -C samples/mnist_conv release
	@+make -C samples/cifar10_conv release

debug:
	@+make -C core debug
	@+make -C tests debug
	@+make -C samples/toySample debug
	@+make -C samples/mnist debug
	@+make -C samples/mnist_conv debug
	@+make -C samples/cifar10_conv debug

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

tidy:
	@+make -C core tidy
	@+make -C samples/toySample tidy
	@+make -C samples/mnist tidy
	@+make -C samples/mnist_conv tidy

help:
	@echo "Possible commands:"
	@echo "\tall     - release + debug"
	@echo "\trelease - release version of library"
	@echo "\tdebug   - debug version of library"
	@echo "\tclean   - cleans all files"
	@echo "\tctags   - creates tags for all files"
	@echo "\tformat  - runs clang-format for all source files"
