from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "memetic.insertion.feasibility_check",
        ["memetic/insertion/feasibility_check.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-O3']  # Optimization
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"})
)