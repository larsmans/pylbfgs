from nose.tools import assert_equal, assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal

from lbfgs import LBFGS, LBFGSError, fmin_lbfgs


def test_fmin_lbfgs():
    def f(x, g, *args):
        g[0] = 2 * x
        return x ** 2

    xmin = fmin_lbfgs(f, 100.)
    assert_array_equal(xmin, [0])


def test_2d():
    def f(x, g, *args):
        assert_equal(x.shape, (2, 2))
        assert_equal(g.shape, x.shape)
        g[:] = 2 * x
        return (x ** 2).sum()

    xmin = fmin_lbfgs(f, [[10., 100.], [44., 55.]])
    assert_array_almost_equal(xmin, [[0, 0], [0, 0]])


def test_class_interface():
    def f(x, g, *args):
        g[:] =  4 * x
        return x ** 4 + 1

    opt = LBFGS()
    opt.max_iterations = 3

    assert_array_equal(opt.minimize(f, 1e6), [1])

    opt.max_iterations = 1
    assert_raises(LBFGSError, opt.minimize, f, 1e7)
