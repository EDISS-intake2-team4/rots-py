from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "optim",
        sources=["optim.pyx"],
        include_dirs=[np.get_include()]  # Specify the NumPy include directory
    )
]

setup(
    ext_modules=cythonize(extensions,        
    compiler_directives = { "language_level" : "3"}),
)
# setup(
#     ext_modules=cythonize("optim.pyx",        
#         compiler_directives = { "language_level" : "3"}),
# )