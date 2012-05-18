all: lbfgs.so

clean:
	rm -rf build
	rm -f 'lbfgs/_lowlevel.{c,o,so}'

install: all
	python setup.py install

inplace: lbfgs/_lowlevel.pyx
	python setup.py build_ext -i
