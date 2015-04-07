from nose.tools import assert_equal, assert_greater, assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal

from lbfgs import LBFGS, LBFGSError, fmin_lbfgs
import numpy as np


def test_fmin_lbfgs():
    def f(x, g, *args):
        g[0] = 2 * x
        return x ** 2

    xmin = fmin_lbfgs(f, 100.)
    assert_array_equal(xmin, [0])


def test_2d():
    def f(x, g, f_calls):
        #f_calls, = args
        assert_equal(x.shape, (2, 2))
        assert_equal(g.shape, x.shape)
        g[:] = 2 * x
        f_calls[0] += 1
        return (x ** 2).sum()

    def progress(x, g, fx, xnorm, gnorm, step, k, ls, p_calls):
        assert_equal(x.shape, (2, 2))
        assert_equal(g.shape, x.shape)

        assert_equal(np.sqrt((x ** 2).sum()), xnorm)
        assert_equal(np.sqrt((g ** 2).sum()), gnorm)

        p_calls[0] += 1
        return 0

    f_calls = [0]
    p_calls = [0]

    xmin = fmin_lbfgs(f, [[10., 100.], [44., 55.]], progress, args=[f_calls])
    assert_greater(f_calls, 0)
    assert_greater(p_calls, 0)
    assert_array_almost_equal(xmin, [[0, 0], [0, 0]])


def test_class_interface():
    def f(x, g, *args):
        g[:] =  4 * x
        return x ** 4 + 1

    opt = LBFGS()
    opt.max_iterations = 3

    assert_array_equal(opt.minimize(f, 1e6), [0])

    opt.max_iterations = 1
    assert_raises(LBFGSError, opt.minimize, f, 1e7)


def test_input_validation():
    assert_raises(TypeError, fmin_lbfgs, [], 1e4)
    assert_raises(TypeError, fmin_lbfgs, lambda x: x, 1e4, "ham")
    assert_raises(TypeError, fmin_lbfgs, lambda x: x, "spam")
