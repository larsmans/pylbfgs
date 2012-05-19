PyLBFGS
=======

This is a Python wrapper around Naoaki Okazaki (chokkan)'s liblbfgs_ library
of quasi-Newton optimization routines (limited memory BFGS and OWL-QN).

This package aims to provide a cleaner interface to the LBFGS algorithm
than is currently available in SciPy_,
and to provide the OWL-QN algorithm to Python users.

To build PyLBFGS, run ``python setup.py build_ext``.


Installing
==========
PyLBFGS is written in Cython_ and requires setuptools_, NumPy_, liblbfgs and
a relatively recent Cython compiler to build (tested with 0.15.1).

Type::

    python setup.py install

(optionally prefixed with ``sudo``) to build and install PyLBFGS.


Hacking
=======
Type::

    python setup.py build_ext -i

to build PyLBFGS in-place, i.e. without installing it.

To run the test suite, make sure you have Nose_ installed, and type::

    nosetests tests/


Authors
=======
PyLBFGS was written by Lars Buitinck.

Alexis Mignon submitted a patch for error handling.


.. _Cython: http://cython.org/

.. _liblbfgs: http://chokkan.org/software/liblbfgs/

.. _Nose: http://readthedocs.org/docs/nose/

.. _NumPy: http://numpy.scipy.org/

.. _SciPy: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html

.. _setuptools: http://pypi.python.org/pypi/setuptools
