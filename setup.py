from setuptools import setup, find_packages, Extension
from Cython.Distutils import build_ext
import numpy

# Define the Cython extension module
cython_extension = Extension(
    "src.datatypes.graphops.graphops_c",
    ["src/datatypes/graphops/graphops_c.pyx"]
)

# # Define the C++ extension module
# cpp_extension = Extension(
#     "src.evaluation.utils.orca.orca",
#     sources=["src/test/utils/orca/orca.cpp"],
#     extra_compile_args=["-O2", "-std=c++11"]
# )

# Specify both extension modules
#ext_modules = [cython_extension, cpp_extension]
ext_modules = [cython_extension]

# Setup the package
setup(
    name='graph-generation',
    version="1.1.0",
    packages=find_packages(),
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
    zip_safe=False,
    include_dirs=[numpy.get_include()]
)

# run command g++ -O2 -std=c++11 -o ./src/test/utils/orca/orca ./src/test/utils/orca/orca.cpp
# to compile the c++ code
import os

os.system("g++ -O2 -std=c++11 -o ./src/evaluation/utils/orca/orca ./src/evaluation/utils/orca/orca.cpp")