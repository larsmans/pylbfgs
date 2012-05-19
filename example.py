"""Trivial example: minimize x**2 from any start value"""

import lbfgs
import sys


def f(x, g):
    """Returns x**2 and stores its gradient in g[0]"""
    x = x[0]
    g[0] = 2*x
    return x**2


def progress(x, g, f_x, xnorm, gnorm, step, k, ls):
    """Report optimization progress."""
    print("x = %8.2g     f(x) = %8.2g     f'(x) = %8.2g" % (x, f_x, g))


try:
    x0 = float(sys.argv[1])
except IndexError:
    print("usage: python %s start-value" % sys.argv[0])
    sys.exit(1)

print("Minimum found: %f" % lbfgs.fmin_lbfgs(f, x0, progress)[0])
