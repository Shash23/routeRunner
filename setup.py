"""Build Cython extension for route_builder hot path."""
from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension(
    "route_builder.hot_path_cy",
    sources=["route_builder/hot_path_cy.pyx"],
)
setup(ext_modules=cythonize(ext, language_level="3"))
