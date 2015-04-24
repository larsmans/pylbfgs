#!/usr/bin/env python
import sys
from setuptools import setup, Extension


# from Michael Hoffman's http://www.ebi.ac.uk/~hoffman/software/sunflower/
class NumpyExtension(Extension):

    def __init__(self, *args, **kwargs):
        Extension.__init__(self, *args, **kwargs)

        self._include_dirs = self.include_dirs
        del self.include_dirs  # restore overwritten property

    # warning: Extension is a classic class so it's not really read-only

    def get_include_dirs(self):
        from numpy import get_include

        return self._include_dirs + [get_include()]

    def set_include_dirs(self, value):
        self._include_dirs = value

    def del_include_dirs(self):
        pass
        
    include_dirs = property(get_include_dirs, 
                            set_include_dirs, 
                            del_include_dirs)

include_dirs = ['liblbfgs']

if sys.platform == 'win32':
    include_dirs.append('compat/win32')

setup(
    name="PyLBFGS",
    version="0.1.6",
    description="LBFGS and OWL-QN optimization algorithms",
    author="Lars Buitinck, Forest Gregg",
    author_email="fgregg@gmail.com",
    packages=['lbfgs'],
    install_requires=['numpy'],
    ext_modules=[NumpyExtension('lbfgs._lowlevel', 
                                ['lbfgs/_lowlevel.c', 'liblbfgs/lbfgs.c'],
                                include_dirs=include_dirs)],
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
)

