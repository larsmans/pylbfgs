PyLBFGS
=======

.. image:: https://travis-ci.org/datamade/pylbfgs.svg?branch=master
    :target: https://travis-ci.org/datamade/pylbfgs


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

    pip install -r requirements.txt
    cython lbfgs/_lowlevel.pyx
    python setup.py build_ext -i

to build PyLBFGS in-place, i.e. without installing it.

To run the test suite, make sure you have pytest_ installed, and type::

    pytest tests


Authors
=======
PyLBFGS was written by Lars Buitinck.

Alexis Mignon submitted a patch for error handling.


.. _Cython: http://cython.org/

.. _liblbfgs: http://chokkan.org/software/liblbfgs/

.. _pytest: http://doc.pytest.org/en/latest/

.. _NumPy: http://numpy.scipy.org/

.. _SciPy: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html

.. _setuptools: http://pypi.python.org/pypi/setuptools
