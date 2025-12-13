import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "neuralnet",
        ["cppfiles/bindings.cpp", "cppfiles/NeuralNet.cpp"],
        include_dirs=[pybind11.get_include()],
        cxx_std=17,
    ),
]

setup(
    name="neuralnet",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
