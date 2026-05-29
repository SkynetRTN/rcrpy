"""Sphinx configuration for the rcrpy documentation.

Built locally with::

    uv pip install -e ".[docs]"        # from the rcrpy/ directory
    sphinx-build -b html docs docs/_build/html

and on Read the Docs via ../.readthedocs.yaml (repo root).
"""

from __future__ import annotations

import importlib.metadata
import os
import sys

# Make the package importable for autodoc even when it is not installed
# (e.g. a bare local checkout). On Read the Docs the package is pip-installed,
# so this is just a belt-and-braces fallback.
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information ------------------------------------------------------

project = "rcrpy"
author = "Reece Clark and Ruide Fu"
copyright = "2026, RCR authors"

try:
    release = importlib.metadata.version("rcrpy")
except importlib.metadata.PackageNotFoundError:
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",      # Google / NumPy style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",              # lets Sphinx render the existing Markdown docs
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # Not part of the rendered site:
    "README.md",                      # describes the docs/ folder itself
    "full_rcr_handoff_explainer.txt",
]

# -- HTML output --------------------------------------------------------------

html_theme = "furo"
html_title = f"rcrpy {release}"
