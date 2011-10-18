"""
LBFGS and OWL-QN optimization algorithms

Python wrapper around liblbfgs.

Written by Lars Buitinck.
Copyright 2011 University of Amsterdam
License: 3-clause BSD
"""

cimport numpy as np
import numpy as np


np.import_array()   # initialize Numpy


cdef extern from "lbfgs.h":
    ctypedef double lbfgsfloatval_t
    ctypedef lbfgsfloatval_t* lbfgsconst_p "const lbfgsfloatval_t *"

    ctypedef lbfgsfloatval_t (*lbfgs_evaluate_t)(void *, lbfgsconst_p, lbfgsfloatval_t *, int, lbfgsfloatval_t)
    ctypedef int (*lbfgs_progress_t)(void *, lbfgsconst_p, lbfgsconst_p, lbfgsfloatval_t, lbfgsfloatval_t, lbfgsfloatval_t, lbfgsfloatval_t, int, int, int)

    cdef enum LineSearchAlgo:
        LBFGS_LINESEARCH_DEFAULT,
        LBFGS_LINESEARCH_MORETHUENTE,
        LBFGS_LINESEARCH_BACKTRACKING_ARMIJO,
        LBFGS_LINESEARCH_BACKTRACKING_WOLFE,
        LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE


    ctypedef struct lbfgs_parameter_t:
        int m
        lbfgsfloatval_t epsilon
        int past
        lbfgsfloatval_t delta
        int max_iterations
        int linesearch
        int max_linesearch
        lbfgsfloatval_t min_step
        lbfgsfloatval_t max_step
        lbfgsfloatval_t ftol
        lbfgsfloatval_t wolfe
        lbfgsfloatval_t gtol
        lbfgsfloatval_t xtol
        lbfgsfloatval_t orthantwise_c
        int orthantwise_start
        int orthantwise_end

    int lbfgs(int, lbfgsfloatval_t *, lbfgsfloatval_t *, lbfgs_evaluate_t,
              lbfgs_progress_t, void *, lbfgs_parameter_t *)

    void lbfgs_parameter_init(lbfgs_parameter_t *)
    lbfgsfloatval_t *lbfgs_malloc(int)
    void lbfgs_free(lbfgsfloatval_t *)


cdef class CallbackData(object):
    cdef object eval_fn
    cdef object progress_fn
    cdef object extra

    def __init__(self, eval_fn, progress_fn, extra):
        self.eval_fn = eval_fn
        self.progress_fn = progress_fn
        self.extra = extra


# Callback into Python evaluation callable.
cdef lbfgsfloatval_t call_eval(void *cb_data_v,
                               lbfgsconst_p x, lbfgsfloatval_t *g,
                               int n, lbfgsfloatval_t step):
    cdef object cb_data
    cdef np.npy_intp shape[1]

    callback_data = <object>cb_data_v
    (f, progress_fn, args) = callback_data
    shape[0] = <np.npy_intp>n
    x_array = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, <void *>x)
    g_array = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, <void *>g)

    return f(x_array, g_array)


# Callback into Python progress reporting callable.
cdef int call_progress(void *cb_data_v,
                       lbfgsconst_p x, lbfgsconst_p g,
                       lbfgsfloatval_t fx,
                       lbfgsfloatval_t xnorm, lbfgsfloatval_t gnorm,
                       lbfgsfloatval_t step, int n, int k, int ls):
    cdef object cb_data
    cdef np.npy_intp shape[1]

    callback_data = <object>cb_data_v
    (f, progress_fn, args) = callback_data

    if progress_fn:
        shape[0] = <np.npy_intp>n
        x_array = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, <void *>x)
        g_array = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, <void *>g)

        return progress_fn(x_array, g_array, fx, xnorm, gnorm, step, k, ls)
    else:
        return 0


# Copy an ndarray to a buffer allocated with lbfgs_malloc; needed to get the
# alignment right for SSE instructions.
cdef lbfgsfloatval_t *copy_to_lbfgs(x):
    x = np.asanyarray(x)
    n = x.shape[0]
    x_copy = lbfgs_malloc(n)
    if x_copy is NULL:
        raise MemoryError
    for i in xrange(n):
        x_copy[i] = x[i]
    return x_copy


LINE_SEARCH_ALGO = {
    'default' : LBFGS_LINESEARCH_DEFAULT,
    'morethuente' : LBFGS_LINESEARCH_MORETHUENTE,
    'armijo' : LBFGS_LINESEARCH_BACKTRACKING_ARMIJO,
    'wolfe' : LBFGS_LINESEARCH_BACKTRACKING_WOLFE,
    'strongwolfe' : LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE
}


cdef class LBFGS(object):
    """LBFGS algorithm, wrapped in a class to permit setting parameters"""

    cdef lbfgs_parameter_t params

    def __init__(self):
        lbfgs_parameter_init(&self.params)

    LINE_SEARCH_ALGORITHMS = LINE_SEARCH_ALGO.keys()

    property m:
        def __set__(self, int val):
            self.params.m = val

    property epsilon:
        def __set__(self, double val):
            self.params.epsilon = val

    property past:
        def __set__(self, int val):
            self.params.past = val

    property delta:
        def __set__(self, double val):
            self.params.delta = val

    property max_iterations:
        def __set__(self, int val):
            self.params.max_iterations = val

    property linesearch:
        def __set__(self, algorithm):
            self.params.linesearch = LINE_SEARCH_ALGO[algorithm]

    property min_step:
        def __set__(self, double val):
            self.params.min_step = val

    property max_step:
        def __set__(self, double val):
            self.params.max_step = val

    property ftol:
        def __set__(self, double val):
            self.params.ftol = val

    property gtol:
        def __set__(self, double val):
            self.params.gtol = val

    property xtol:
        def __set__(self, double val):
            self.params.xtol = val

    property wolfe:
        def __set__(self, double val):
            self.params.wolfe = val

    property orthantwise_c:
        def __set__(self, double val):
            self.params.orthantwise_c = val

    property orthantwise_start:
        def __set__(self, int val):
            self.params.orthantwise_start = val

    property orthantwise_end:
        def __set__(self, int val):
            self.params.orthantwise_end = val


    def minimize(self, f, x0, progress=None, args=()):
        """Minimize a function using LBFGS or OWL-QN

        Parameters
        ----------
        f : callable(x, g, *args)
            Computes function and gradient of function to minimize. Called with the
            current position vector x, a vector g and *args; must return the value
            f(x) and set the gradient vector g.
        x0 : array-like
            Initial values. A copy of this array is made prior to optimization.
        progress : callable(x, g, fx, xnorm, gnorm, step, k, ls)
        """

        cdef int n
        cdef lbfgsfloatval_t *x
        cdef np.ndarray x_final

        x = copy_to_lbfgs(x0)
        n = x0.shape[0]

        x_final = np.empty(n, dtype=np.double)

        callback_data = (f, progress, args)
        r = lbfgs(n, x, <lbfgsfloatval_t *>x_final.data, call_eval,
                  call_progress, <void *>callback_data, &self.params)

        if r == 0:
            return x_final


def fmin_lbfgs(f, x0, progress=None, args=()):
    return LBFGS().minimize(f, x0, progress=progress, args=args)
