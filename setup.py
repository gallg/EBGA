from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="EBGA",
    version="0.2.3",
    author="Giuseppe Gallitto",
    description="Evolutionary-Based Generative Adaptation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://codeberg.org/Nevdyf/EBGA",
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={
        "": ["LICENSE"],
    },
    classifiers=[
        "Development Status :: Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=2.4.6",
        "scikit-learn>=1.8.0"
    ],
    extras_require={
        "dev": [
            "mkdocs>=1.6.1"
        ],
    },
)
