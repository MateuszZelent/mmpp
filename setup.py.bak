from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="mmpp",
    version="0.5.3",
    author="Mateusz Zelent",
    author_email="mateusz.zelent@amu.edu.pl",
    description="A library for mmpp (Micro Magnetic Post Processing) simulation and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mateuszzelent/mmpp",  # Dodaj URL do repo
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "pyzfn",
        "zarr",
        "rich",
        "tqdm",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "ruff",
            "mypy",
            "build",
            "twine",
        ],
        "interactive": [
            "itables",
            "IPython",
            "jupyter",
        ],
        "plotting": [
            "cmocean",
        ],
    },
    include_package_data=True,
    package_data={
        "mmpp": [
            "paper.mplstyle",
            "fonts/**/*",
        ],
    },
    entry_points={
        "console_scripts": [
            "mmpp=mmpp.cli:main",
        ],
    },
)
