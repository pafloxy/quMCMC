from setuptools import setup

from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="qumcmc",
    version="0.0.1",
    description="Implementation of QAOA for Travelling Salesman Problem",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Neelkanth Rawat, Rajarsi Pal, Manuel S. Rudolph",
    author_email="rajarsi14oc@gmail.com",
    url="https://github.com/pafloxy/quMCMC",
    python_requires=">=3.7",
    packages=["qumcmc"],
    install_requires=[
        "qiskit>=0.29.0",
        "qiskit-machine-learning",
        "qiskit-optimization",
        "pandas",
        "networkx",
        "numpy",
        "pytest",
        "tqdm",
        "cvxgraphalgs",
        "cvxopt",
        "scikit-learn==1.0",
        "notebook",
        "matplotlib",
        "seaborn",
        "itertools",
        "collections",
        "typing",
        "math"

    ],
    zip_safe=True,
)