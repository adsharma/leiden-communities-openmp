import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "leiden_communities._leiden_cpp",
        [
            "src/python_bindings.cpp",  # Main binding file
        ],
        include_dirs=[
            # Path to pybind11 headers
            pybind11.get_include(),
            # Path to project headers
            "inc",
        ],
        language="c++",
        cxx_std=17,
        define_macros=[
            ("OPENMP", None),
            ("TYPE", "float"),
            ("MAX_THREADS", "64"),
        ],
        extra_compile_args=[
            "-fopenmp",
            "-O3",
            "-march=native",
            "-ffast-math",
        ],
        extra_link_args=[
            "-fopenmp",
        ],
    ),
]

setup(
    name="leiden-communities-openmp",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="High-performance Leiden and Louvain community detection with pandas/Arrow support",
    long_description="",
    packages=["leiden_communities"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.0.0",
        "pyarrow>=5.0.0",
        "numpy>=1.19.0",
        "pybind11>=2.6.0",
    ],
)
