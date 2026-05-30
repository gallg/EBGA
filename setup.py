from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="EBGA",
    version="0.1.0",
    author="EBGA Team",
    author_email="",
    description="Evolutionary-Based Gradient-free Architecture - A gradient-free neural network framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/EBGA",
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={
        "": ["LICENSE"],
    },
    classifiers=[
        "Development Status :: 3 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "mkdocs>=1.0.0",
            "pytest>=7.0.0",
        ],
    },
)
