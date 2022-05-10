from setuptools import find_packages, setup


setup(
    name="mytorch",
    version="0.0.0",
    description="pytorch-lightning extensions",
    long_description="",
    author="Yuki Imajuku",
    license="Apache License 2.0",
    url="https://github.com/Yuki-Imajuku/mytorch",
    packages=find_packages(),
    python_requires=">=3.7",
    zip_safe=False,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
