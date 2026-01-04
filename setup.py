from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of the README file for the long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="neuralsutra",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch",
        "sympy",
        "numpy",
        "scikit-learn",
        "pytest",
    ],
    # Metadata for recruiters/users
    description="A Hybrid Neuro-Symbolic Integration Engine using Vedic Mathematics Kernels",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sai Nikhil Tadepalli",
    url="https://github.com/SaiNikhilTadepalli/NeuralSutra",
    project_urls={
        "Bug Tracker": "https://github.com/SaiNikhilTadepalli/NeuralSutra/issues",
        "Documentation": "https://github.com/SaiNikhilTadepalli/NeuralSutra#readme",
    },
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.11",
)
