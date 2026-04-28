# Installation

`autoray` is available on both [pypi](https://pypi.org/project/autoray/) and
[conda-forge](https://anaconda.org/conda-forge/autoray). While `autoray` is
pure python and has no direct dependencies itself, the preferred way to install
it is with [pixi](https://pixi.sh), which creates isolated and reproducible
environments that can mix packages from [`conda-forge`](https://conda-forge.org/) (the default) and also [`pypi`](https://pypi.org/).

**Installing with `pixi` (preferred):**
```bash
pixi init autoray-project
cd autoray-project
pixi add autoray
```

**Installing with `pip`:**
```bash
pip install autoray
# or
uv pip install autoray
```
It is recommended to use [`uv`](https://docs.astral.sh/uv/) to install and
manage purely pypi based environments.

**Installing with `conda` / `mamba`:**
```bash
conda install -c conda-forge autoray
```
[`miniforge`](https://github.com/conda-forge/miniforge) is the recommended way
to manage and install a conda-based environment.


**Installing the latest version directly from github:**

If you want to checkout the latest version of features and fixes, you can
install directly from the github repository:
```bash
pip install -U git+https://github.com/jcmgray/autoray.git
```

**Installing a local, editable development version:**

If you want to make changes to the source code and test them out, you can
install a local editable version of the package:
```bash
git clone https://github.com/jcmgray/autoray.git
pip install --no-deps -U -e autoray/
```

```{note}
**No-install version:**
The entirety of the automatic dispatch mechanism is contained in the single
file `autoray.py`, which you could simply copy into your project if you don't
want add a dependency.
```

## Optional plotting requirements

The [`autoray.lazy.draw`](autoray.lazy.draw) visualizations variously require:

* [`matplotlib`](https://matplotlib.org/)
* [`networkx`](https://networkx.org/) - for computational graph drawing
* [`pygraphviz`](https://pygraphviz.github.io/) - optional, for better and
  faster graph layouts than `networkx`.
