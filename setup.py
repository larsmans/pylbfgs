#!/usr/bin/env python

from distutils.core import setup
from distutils.command.build_clib import build_clib
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [
    Extension("lbfgs._lowlevel", ["lbfgs/_lowlevel.pyx", "liblbfgs/lbfgs.c"],
              include_dirs=[np.get_include(), 'liblbfgs']),
]

setup(
    name="PyLBFGS",
    version="0.0",
    description="LBFGS and OWL-QN optimization algorithms",
    author="Lars Buitinck",
    author_email="L.J.Buitinck@uva.nl",
    packages=['lbfgs'],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Programming Language :: Cython",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    cmdclass={"build_clib" : build_clib, "build_ext": build_ext},
    ext_modules=ext_modules
)

