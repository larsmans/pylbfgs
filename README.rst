PyLBFGS
=======

This is a Python wrapper around Naoaki Okazaki (chokkan)'s liblbfgs_ library
of quasi-Newton optimization routines (limited memory BFGS and OWL-QN).

This package aims to provide a cleaner interface to the LBFGS
algorithm than is currently available in SciPy_, and to provide the
OWL-QN algorithm to Python users.


Installing
==========
Type::

    pip install pylbfgs


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
