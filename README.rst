PyLBFGS
=======

This is a Python wrapper around Naoaki Okazaki (chokkan)'s liblbfgs_ library
of quasi-Newton optimization routines (limited memory BFGS and OWL-QN).
It is written in Cython_ and currently requires
a relatively recent Cython compiler to properly build.

This package exists to provide a lower-level, but cleaner interface
to the LBFGS algorithm than is currently available in SciPy_,
and to provide the OWL-QN algorithm to Python users.

To build PyLBFGS, run ``python setup.py build_ext``.


.. _Cython: http://cython.org

.. _liblbfgs: http://chokkan.org/software/liblbfgs/

.. _SciPy: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
