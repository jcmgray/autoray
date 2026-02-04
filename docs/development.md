# Development


## Contributing


## Testing


## Building the documentation


## Minting a release

`autoray` uses
[`setuptools_scm`](https://pypi.org/project/setuptools-scm/) to manage versions
and [github actions](https://github.com/jcmgray/autoray/actions) to
automatically publish to [PyPI](https://pypi.org/project/autoray/). To mint a
new release:

1. Make sure all the [tests are passing on CI](https://github.com/jcmgray/autoray/actions/workflows/tests.yml)
2. Tag the version like `vX.X.X` (e.g. `v1.2.3`)
3. Push the tag to github, which will trigger building and uploading a package
   to the [PyPI **test** server](https://test.pypi.org/project/autoray/).
4. If all goes well, create a release on github and publish to trigger building
   and uploading a package to the [PyPI **production** server](https://pypi.org/project/autoray/).
5. The [`conda-forge/autoray-feedstock`](https://github.com/conda-forge/autoray-feedstock)
   repo should automatically pick up the new PyPI release and build a new
   [conda package](https://anaconda.org/conda-forge/autoray), the recipe should
   only need to be manually updated if there are for example new dependencies.
