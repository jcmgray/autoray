# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.append(os.path.abspath("./_pygments"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'autoray'
copyright = '2019-2023, Johnnie Gray'
author = 'Johnnie Gray'

try:
    from autoray import __version__
    release = __version__
except ImportError:
    try:
        from importlib.metadata import version as _version
        release = _version('autoray')
    except ImportError:
        release = '0.0.0+unknown'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_nb',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
    'sphinx.ext.napoleon',
    'sphinx.ext.linkcode',
    'sphinx_copybutton',
    'autoapi.extension',
]

nb_execution_mode = "off"
myst_heading_anchors = 4
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

# sphinx-autoapi
autoapi_dirs = ['../autoray']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'

html_theme_options = {
    "sidebar_hide_name": True,
    # "light_css_variables": {
    #     "color-brand-primary": "hsl(72, 75%, 40%)",
    #     "color-brand-content": "hsl(238, 50%, 60%)",
    # },
    # "dark_css_variables": {
    #     "color-brand-primary": "hsl(72, 75%, 60%)",
    #     "color-brand-content": "hsl(238, 75%, 70%)",
    # },
    "light_logo": "autoray-header.png",
    "dark_logo": "autoray-header.png",
}

html_css_files = ["my-styles.css"]
html_static_path = ['_static']
# html_favicon = "_static/logo-favicon.ico"

pygments_style = '_pygments_light.MarianaLight'
pygments_dark_style = "_pygments_dark.MarianaDark"


def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object
    """
    import autoray
    import inspect

    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    fn = os.path.relpath(fn, start=os.path.dirname(autoray.__file__))

    if "+" in autoray.__version__:
        return (
            f"https://github.com/jcmgray/autoray/blob/"
            f"HEAD/autoray/{fn}{linespec}"
        )
    else:
        return (
            f"https://github.com/jcmgray/autoray/blob/"
            f"v{autoray.__version__}/autoray/{fn}{linespec}"
        )
