from numpy.testing import assert_array_equal

from lbfgs import fmin_lbfgs


def test_fmin_lbfgs():
    def f(x, g, *args):
        g[0] = 2 * x
        return x ** 2

    xmin = fmin_lbfgs(f, 100.)
    assert_array_equal(xmin, [0])
