# Development


## Contributing


## Testing


## Building the documentation


## Minting a release

`autoray` uses `setuptools_scm` to manage versions and github actions to
automatically publish to PyPI. To mint a new release:

1. Make sure all the tests are passing on CI
2. Tag the version like `vX.X.X` (e.g. `v1.2.3`)
3. Push the tag to github, which will trigger building and uploading a package
   to the PyPI **test** server.
4. If all goes well, create a release on github and publish to trigger building
   and uploading a package to the PyPI **production** server.
5. The `conda-forge/autoray-feedstock` repo should automatically pick up the
   new PyPI release and build a new conda package.
