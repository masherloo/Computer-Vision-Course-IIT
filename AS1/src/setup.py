from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name = 'c2D',
    ext_modules = cythonize("convolution2D.pyx"),
    include_dirs=[numpy.get_include()]
)