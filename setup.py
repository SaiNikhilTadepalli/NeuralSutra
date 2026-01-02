from setuptools import setup, find_packages

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
    description="A Neuro-Symbolic Vedic Integration Engine",
    author="Sai Nikhil Tadepalli",
    python_requires=">=3.11",
)
