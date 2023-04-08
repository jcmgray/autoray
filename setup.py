from setuptools import setup, find_packages


def readme():
    with open("README.md") as f:
        long_desc = f.read()
        # strip out the raw html images?
        return long_desc


short_desc = "Abstract your array operations."

setup(
    name="autoray",
    description=short_desc,
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="http://github.com/jcmgray/autoray",
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/jcmgray/autoray/issues',
        'Source': 'https://github.com/jcmgray/autoray/',
    },
    author="Johnnie Gray",
    author_email="johnniemcgray@gmail.com",
    license="Apache",
    packages=find_packages(exclude=["deps", "tests*"]),
    extras_require={
        "tests": [
            "numpy",
            "coverage",
            "pytest",
            "pytest-cov",
        ],
        'docs': [
            'sphinx>=2.0',
            'sphinx-autoapi',
            'sphinx-copybutton',
            'myst-nb',
            'furo',
            'setuptools_scm',
            'ipython!=8.7.0',
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="array agnostic numeric numpy cupy dask tensorflow jax autograd",
)
