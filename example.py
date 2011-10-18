"""Trivial example: minimize x**2 from any start value"""

import lbfgs
import numpy as np
import sys

def f(x, g):
    """Returns x**2 and stores its gradient in g[0]"""
    x = x[0]
    g[0] = 2*x
    return x**2

x0 = np.asarray([float(sys.argv[1])])
print lbfgs.fmin_lbfgs(f, x0)[0]
