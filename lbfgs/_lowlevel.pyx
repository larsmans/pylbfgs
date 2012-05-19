"""
LBFGS and OWL-QN optimization algorithms

Python wrapper around liblbfgs.
"""

cimport numpy as np
import numpy as np


np.import_array()   # initialize Numpy


cdef extern from "lbfgs.h":
    ctypedef double lbfgsfloatval_t
    ctypedef lbfgsfloatval_t* lbfgsconst_p "const lbfgsfloatval_t *"

    ctypedef lbfgsfloatval_t (*lbfgs_evaluate_t)(void *, lbfgsconst_p,
                              lbfgsfloatval_t *, int, lbfgsfloatval_t)
    ctypedef int (*lbfgs_progress_t)(void *, lbfgsconst_p, lbfgsconst_p,
                                     lbfgsfloatval_t, lbfgsfloatval_t,
                                     lbfgsfloatval_t, lbfgsfloatval_t,
                                     int, int, int)

    cdef enum LineSearchAlgo:
        LBFGS_LINESEARCH_DEFAULT,
        LBFGS_LINESEARCH_MORETHUENTE,
        LBFGS_LINESEARCH_BACKTRACKING_ARMIJO,
        LBFGS_LINESEARCH_BACKTRACKING_WOLFE,
        LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE

    cdef enum ReturnCode:
        LBFGS_SUCCESS,
        LBFGS_ALREADY_MINIMIZED,
        LBFGSERR_UNKNOWNERROR,
        LBFGSERR_LOGICERROR,
        LBFGSERR_OUTOFMEMORY,
        LBFGSERR_CANCELED,
        LBFGSERR_INVALID_N,
        LBFGSERR_INVALID_N_SSE,
        LBFGSERR_INVALID_X_SSE,
        LBFGSERR_INVALID_EPSILON,
        LBFGSERR_INVALID_TESTPERIOD,
        LBFGSERR_INVALID_DELTA,
        LBFGSERR_INVALID_LINESEARCH,
        LBFGSERR_INVALID_MINSTEP,
        LBFGSERR_INVALID_MAXSTEP,
        LBFGSERR_INVALID_FTOL,
        LBFGSERR_INVALID_WOLFE,
        LBFGSERR_INVALID_GTOL,
        LBFGSERR_INVALID_XTOL,
        LBFGSERR_INVALID_MAXLINESEARCH,
        LBFGSERR_INVALID_ORTHANTWISE,
        LBFGSERR_INVALID_ORTHANTWISE_START,
        LBFGSERR_INVALID_ORTHANTWISE_END,
        LBFGSERR_OUTOFINTERVAL,
        LBFGSERR_INCORRECT_TMINMAX,
        LBFGSERR_ROUNDING_ERROR,
        LBFGSERR_MINIMUMSTEP,
        LBFGSERR_MAXIMUMSTEP,
        LBFGSERR_MAXIMUMLINESEARCH,
        LBFGSERR_MAXIMUMITERATION,
        LBFGSERR_WIDTHTOOSMALL,
        LBFGSERR_INVALIDPARAMETERS,
        LBFGSERR_INCREASEGRADIENT

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
    cdef np.npy_intp tshape[1]

    callback_data = <object>cb_data_v
    (f, progress_fn, shape, args) = callback_data
    tshape[0] = <np.npy_intp>n
    x_array = np.PyArray_SimpleNewFromData(1, tshape, np.NPY_DOUBLE, <void *>x)
    g_array = np.PyArray_SimpleNewFromData(1, tshape, np.NPY_DOUBLE, <void *>g)

    return f(x_array.reshape(shape), g_array.reshape(shape), *args)


# Callback into Python progress reporting callable.
cdef int call_progress(void *cb_data_v,
                       lbfgsconst_p x, lbfgsconst_p g,
                       lbfgsfloatval_t fx,
                       lbfgsfloatval_t xnorm, lbfgsfloatval_t gnorm,
                       lbfgsfloatval_t step, int n, int k, int ls):
    cdef object cb_data
    cdef np.npy_intp tshape[1]

    callback_data = <object>cb_data_v
    (f, progress_fn, shape, args) = callback_data

    if progress_fn:
        tshape[0] = <np.npy_intp>n
        x_array = np.PyArray_SimpleNewFromData(1, tshape, np.NPY_DOUBLE,
                                               <void *>x)
        g_array = np.PyArray_SimpleNewFromData(1, tshape, np.NPY_DOUBLE,
                                               <void *>g)

        r = progress_fn(x_array.reshape(shape), g_array.reshape(shape), fx,
                        xnorm, gnorm, step, k, ls)
        # TODO what happens when the callback returns the wrong type?
        return 0 if r is None else r
    else:
        return 0


# Copy an ndarray to a buffer allocated with lbfgs_malloc; needed to get the
# alignment right for SSE instructions.
cdef lbfgsfloatval_t *copy_to_lbfgs(x):
    n = x.shape[0]
    x_copy = lbfgs_malloc(n)
    if x_copy is NULL:
        raise MemoryError
    for i in xrange(n):
        x_copy[i] = x[i]
    return x_copy


_LINE_SEARCH_ALGO = {
    'default' : LBFGS_LINESEARCH_DEFAULT,
    'morethuente' : LBFGS_LINESEARCH_MORETHUENTE,
    'armijo' : LBFGS_LINESEARCH_BACKTRACKING_ARMIJO,
    'wolfe' : LBFGS_LINESEARCH_BACKTRACKING_WOLFE,
    'strongwolfe' : LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE
}


_ERROR_MESSAGES = {
    LBFGSERR_UNKNOWNERROR: "Unknown error." ,
    LBFGSERR_LOGICERROR: "Logic error.",
    LBFGSERR_OUTOFMEMORY: "Insufficient memory.",
    LBFGSERR_CANCELED: "The minimization process has been canceled.",
    LBFGSERR_INVALID_N: "Invalid number of variables specified.",
    LBFGSERR_INVALID_N_SSE: "Invalid number of variables (for SSE) specified.",
    LBFGSERR_INVALID_X_SSE: "The array x must be aligned to 16 (for SSE).",
    LBFGSERR_INVALID_EPSILON: "Invalid parameter epsilon specified.",
    LBFGSERR_INVALID_TESTPERIOD: "Invalid parameter past specified.",
    LBFGSERR_INVALID_DELTA: "Invalid parameter delta specified.",
    LBFGSERR_INVALID_LINESEARCH: "Invalid parameter linesearch specified.",
    LBFGSERR_INVALID_MINSTEP: "Invalid parameter max_step specified.",
    LBFGSERR_INVALID_MAXSTEP: "Invalid parameter max_step specified.",
    LBFGSERR_INVALID_FTOL: "Invalid parameter ftol specified.",
    LBFGSERR_INVALID_WOLFE: "Invalid parameter wolfe specified.",
    LBFGSERR_INVALID_GTOL: "Invalid parameter gtol specified.",
    LBFGSERR_INVALID_XTOL: "Invalid parameter xtol specified.",
    LBFGSERR_INVALID_MAXLINESEARCH:
        "Invalid parameter max_linesearch specified.",
    LBFGSERR_INVALID_ORTHANTWISE: "Invalid parameter orthantwise_c specified.",
    LBFGSERR_INVALID_ORTHANTWISE_START:
        "Invalid parameter orthantwise_start specified.",
    LBFGSERR_INVALID_ORTHANTWISE_END:
        "Invalid parameter orthantwise_end specified.",
    LBFGSERR_OUTOFINTERVAL:
        "The line-search step went out of the interval of uncertainty.",
    LBFGSERR_INCORRECT_TMINMAX:
        "A logic error occurred;"
        " alternatively, the interval of uncertainty became too small.",
    LBFGSERR_ROUNDING_ERROR:
        "A rounding error occurred;"
        " alternatively, no line-search step satisfies"
        " the sufficient decrease and curvature conditions.",
    LBFGSERR_MINIMUMSTEP: "The line-search step became smaller than min_step.",
    LBFGSERR_MAXIMUMSTEP: "The line-search step became larger than max_step.",
    LBFGSERR_MAXIMUMLINESEARCH:
        "The line-search routine reaches the maximum number of evaluations.",
    LBFGSERR_MAXIMUMITERATION:
        "The algorithm routine reaches the maximum number of iterations.",
    LBFGSERR_WIDTHTOOSMALL:
        "Relative width of the interval of uncertainty is at most xtol.",
    LBFGSERR_INVALIDPARAMETERS:
        "A logic error (negative line-search step) occurred.",
    LBFGSERR_INCREASEGRADIENT:
        "The current search direction increases the objective function value.",
}


class LBFGSError(Exception):
    pass


cdef class LBFGS(object):
    """LBFGS algorithm, wrapped in a class to permit setting parameters"""

    cdef lbfgs_parameter_t params

    def __init__(self):
        lbfgs_parameter_init(&self.params)

    LINE_SEARCH_ALGORITHMS = _LINE_SEARCH_ALGO.keys()

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
            self.params.linesearch = _LINE_SEARCH_ALGO[algorithm]

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
            Computes function to minimize and its gradient.
            Called with the current position x (a numpy.ndarray), a gradient
            vector g (a numpy.ndarray) to be filled in and *args.
            Must return the value at x and set the gradient vector g.
        x0 : array-like
            Initial values. A copy of this array is made prior to optimization.
        progress : callable(x, g, fx, xnorm, gnorm, step, k, ls)
        args : sequence
            Arbitrary list of arguments, passed on to the function f as *args.
        """

        cdef np.npy_intp n
        cdef int n_i
        cdef int r
        cdef lbfgsfloatval_t *x
        cdef np.ndarray x_final

        x0 = np.atleast_1d(x0)
        x = copy_to_lbfgs(x0.ravel())
        n = np.product(x0.shape)
        n_i = n

        if n_i != n:
            raise LBFGSError("Array of %d elements too large to handle" % n)

        x_final = np.empty(x0.shape, dtype=np.double)

        callback_data = (f, progress, x0.shape, args)
        r = lbfgs(n, x, <lbfgsfloatval_t *>x_final.data, call_eval,
                  call_progress, <void *>callback_data, &self.params)

        if r == LBFGS_SUCCESS or r == LBFGS_ALREADY_MINIMIZED:
            return x_final
        elif r == LBFGSERR_OUTOFMEMORY:
            raise MemoryError
        else:
            raise LBFGSError(_ERROR_MESSAGES[r])
