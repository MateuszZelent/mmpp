from pathlib import Path

from setuptools import find_packages, setup

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
    url="https://github.com/mateuszzelent/mmpp",
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
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "zarr>=2.18.0,<3.0.0; python_version < '3.11'",
        "zarr>=3.0.0; python_version >= '3.11'",
        "h5py>=3.0.0",
        "rich",
        "tqdm",
        "PyYAML>=5.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "ruff",
            "mypy",
            "build",
            "twine",
            "sphinx>=4.0",
            "sphinx-rtd-theme",
            "sphinx-autodoc-typehints",
            "myst-parser",
            "linkify-it-py",
        ],
        "interactive": [
            "itables",
            "IPython",
            "jupyter",
            "ipywidgets",
            "k3d",
            "holoviews",
            "pyvista",
        ],
        "plotting": [
            "cmocean",
            "cmcrameri",
            "k3d",
            "holoviews",
            "pyvista",
        ],
        "fft": [
            "scipy",
            "pyfftw",
        ],
        "wavelets": [
            "PyWavelets",
        ],
        "image": [
            "scikit-image",
            "imageio",
        ],
        "ml": [
            "scikit-learn",
            "joblib",
        ],
        "tui": [
            "textual>=0.40.0",
        ],
        "full": [
            "textual>=0.40.0",
            "cmocean",
            "cmcrameri",
            "k3d",
            "holoviews",
            "pyvista",
            "itables",
            "IPython",
            "jupyter",
            "ipywidgets",
            "scipy",
            "pyfftw",
            "PyWavelets",
            "scikit-image",
            "imageio",
            "scikit-learn",
            "joblib",
            "numba",
            "psutil",
        ],
    },
    include_package_data=True,
    package_data={
        "mmpp": [
            "paper.mplstyle",
            "fonts/**/*",
            "dracula.tcss",
            "pyzfn/**/*.py",
        ],
    },
    entry_points={
        "console_scripts": [
            "mmpp=mmpp.cli:main",
            "mmpp-tui=mmpp.tui:main",
            "mmpp-classic=mmpp.cli.main:main",
        ],
    },
)
