from setuptools import setup, find_packages

setup(
    name="jcmlp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numba>=0.61.0",
        "numpy>=1.26.4"
    ],
)
