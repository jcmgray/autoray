# Installation

`autoray` is available on both [pypi](https://pypi.org/project/autoray/) and
[conda-forge](https://anaconda.org/conda-forge/autoray). While `autoray` is
pure python and has no direct dependencies itself, the recommended distribution
would be [mambaforge](https://github.com/conda-forge/miniforge#mambaforge)
for installing the various backend array libraries and their dependencies.

**Installing with `pip`:**
```bash
pip install autoray
```

**Installing with `conda`:**
```bash
conda install -c conda-forge autoray
```

**Installing with `mambaforge`:**
```bash
mamba install autoray
```
```{hint}
Mamba is a faster version of `conda`, and the -forge distritbution comes
pre-configured with only the `conda-forge` channel, which further simplifies
and speeds up installing dependencies.
```

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
