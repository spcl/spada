#!/usr/bin/env python3
"""Setup script for spatialstencil package."""

from setuptools import setup, find_packages
import os


# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "A spatial stencil compiler for high-performance computing."


# Read version from package
def get_version():
    """Get version from spatialstencil package."""
    try:
        import spatialstencil
        return spatialstencil.__version__
    except (ImportError, AttributeError):
        return "0.1.0"


setup(
    name="spatialstencil",
    version=get_version(),
    author="SpatialStencil Team",
    author_email="",
    description="A spatial stencil compiler for high-performance computing",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/glukas/spatialstencil",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Compilers",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pydot",
        "networkx",
        "lark",
        "matplotlib",
        "hilbertcurve",
        "igraph",
        "click",  # Added for CLI functionality
        "parse",  # From requirements.txt
        "pycairo",  # For rendering graphs
    ],
    extras_require={
        "docs": [
            "mkdocs",
            "mkdocs-material",
            "mkdocs-material-extensions",
            "pymdown-extensions",
        ],
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
        ],
        "all": [
            # Docs dependencies
            "mkdocs",
            "mkdocs-material",
            "mkdocs-material-extensions",
            "pymdown-extensions",
            # Dev dependencies
            "pytest",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
        ],
    },
    entry_points={
        "console_scripts": ["sptlc=spatialstencil.cli.compiler:compile_spatial_ir"],
    },
    include_package_data=True,
    package_data={
        "spatialstencil": ["**/*.py"],
    },
    keywords=[
        "stencil",
        "compiler",
        "spatial",
        "high-performance-computing",
        "csl",
        "cerebras",
    ],
)
