from setuptools import setup, find_packages
import versioneer


def readme():
    with open('README.rst') as f:
        import re
        long_desc = f.read()
        # strip out the raw html images
        long_desc = re.sub('\.\. raw::[\S\s]*?>\n\n', "", long_desc)
        return long_desc


short_desc = ('Write backend agnostic numeric code '
              'compatible with any numpy-ish array library.')

setup(
    name='autoray',
    description=short_desc,
    long_description=readme(),
    url='http://github.com/jcmgray/autoray',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='Johnnie Gray',
    author_email="johnniemcgray@gmail.com",
    license='Apache',
    packages=find_packages(exclude=['deps', 'tests*']),
    install_requires=[
        'numpy',
    ],
    extras_require={
        'tests': [
            'coverage',
            'pytest',
            'pytest-cov',
        ],
    },
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='array agnostic numeric numpy cupy dask tensorflow jax autograd',
)
