all: lbfgs.so

clean:
	rm -rf build
	rm -f lbfgs.{c,o,so}

install: all
	python setup.py install

lbfgs.so: lbfgs.pyx
	python setup.py build_ext
