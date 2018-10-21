include ./makes/defines.inc

all: release debug

release: lib_release test_release
debug: lib_debug test_debug

.PHONY: lib_release lib_debug test_release test_debug clean

#### LIBRARY

lib_release:
	@make -C core release

lib_debug:
	@make -C core debug

#### TESTS

test_release: lib_release
	@make -C tests release

test_debug: lib_debug
	@make -C tests debug

clean:
	rm -r $(OUTDIR)

ctags:
	ctags -R --tag-relative=yes --exclude=.git $(ROOT_DIR)
