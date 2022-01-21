from distutils.core import setup
from setuptools import Extension, setup
from Cython.Build import cythonize

from distutils.extension import Extension

import numpy as np

ext_modules = [
    Extension(
        "interp",
        ["interp3d/interp.pyx"],
        extra_compile_args=['/openmp','/O2'],
#        extra_link_args=['/openmp'],
    )
]

setup(name='interp3d',
      language_level=3,  
      ext_modules=cythonize(ext_modules),
      packages=['interp3d'],
      include_dirs=[np.get_include()])
