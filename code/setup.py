from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extension = Extension(
    "amm_cython",  # Name of the module
    ["amm_cython.pyx"],  # Cython source file
    include_dirs=[numpy.get_include()]  # Adds NumPy headers to include path
)

setup(
    name="AMM Module",
    ext_modules=cythonize(extension)
)
