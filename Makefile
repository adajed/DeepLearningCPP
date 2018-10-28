include ./makes/defines.inc

all: release debug

release:
	@+make -C core release
	@+make -C tests release
	@+make -C samples/toySample release
	@+make -C samples/mnist release

debug:
	@+make -C core debug
	@+make -C tests debug
	@+make -C samples/toySample debug
	@+make -C samples/mnist debug

.PHONY: clean ctags format

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
