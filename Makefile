all: inplace

clean:
	rm -rf build
	rm -f lbfgs/_lowlevel.c
	rm -f lbfgs/_lowlevel.o
	rm -f lbfgs/_lowlevel.so

install: all
	python setup.py install

inplace: lbfgs/_lowlevel.pyx
	python setup.py build_ext -i
