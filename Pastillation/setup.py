from setuptools import setup, find_packages

setup(
    name="pastillation",
    version="0.0.1",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires="==3.8.16",
    install_requires=[
        "tensorflow==2.13.1",
        "numpy==1.24.4",
        "scikit-learn==1.2.2",
        "matplotlib==3.4.3",
        "proplot==0.9.7",
        "scipy=1.9.1",
        "tqdm=4.65.0",
        "imageio==2.34.2",
    ],
)
