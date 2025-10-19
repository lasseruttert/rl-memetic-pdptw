import os
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

ext_modules = [
    Pybind11Extension(
        "memetic.insertion.insertion_core",  # Vollständiger Modulpfad
        ["memetic/insertion/insertion_core.cpp"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=['/O2'] if os.name == 'nt' else ['-O3', '-march=native'],
    ),
    Pybind11Extension(
        "memetic.fitness.fitness_core",  # Vollständiger Modulpfad
        ["memetic/fitness/fitness_core.cpp"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=['/O2'] if os.name == 'nt' else ['-O3', '-march=native'],
    ),
    Pybind11Extension(
        "utils.solution_core",  # Vollständiger Modulpfad
        ["utils/solution_core.cpp"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=['/O2'] if os.name == 'nt' else ['-O3', '-march=native'],
    ),
]

setup(
    name="memetic_cpp_extensions",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)