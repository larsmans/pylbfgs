from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension("lbfgs", ["lbfgs.pyx"],
              libraries = ["lbfgs"])
]

setup(
    name = "PyLBFGS",
    cmdclass = {"build_ext" : build_ext},
    ext_modules = ext_modules
)
