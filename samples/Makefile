include ../makes/defines.inc

SAMPLES = toySample mnist mnist_conv cifar10_conv

release: $(patsubst %,%_release,$(SAMPLES))
debug: $(patsubst %,%_debug,$(SAMPLES))

%_release:
	make -C $* release

%_debug:
	make -C $* debug
