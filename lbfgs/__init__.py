"""
LBFGS and OWL-QN optimization algorithms

Python wrapper around liblbfgs.

Copyright 2011-2012 Lars Buitinck/University of Amsterdam
License: 3-clause BSD
"""

from ._lowlevel import LBFGS, LBFGSError


def fmin_lbfgs(f, x0, progress=None, args=()):
    """Minimize a function using LBFGS or OWL-QN.

    See LBFGS.minimize for full documentation.
    """
    return LBFGS().minimize(f, x0, progress=progress, args=args)
