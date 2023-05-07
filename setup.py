import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ekernels",
    version="0.1.0",
    description="ekernels: Explicit Feature Maps Approximations of Kernels Functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EthanJamesLew/ekernels",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ],
    package_dir={"": ".", "ekernels": "ekernels"},
    install_requires=[
        "numpy>=1.19.0",
    ],
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.8",
)