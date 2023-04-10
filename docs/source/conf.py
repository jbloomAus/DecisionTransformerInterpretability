# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

project = "DecisionTransformerInterpretability"
copyright = "2023, Joseph Bloom"
author = "Joseph Bloom"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinxcontrib.napoleon",
    "myst_parser",
    "sphinx-favicon",
    "sphinx.ext.githubpages",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = []


sys.path.insert(0, os.path.abspath("../.."))

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = "Decision Transformer Interpretability Documentation"
html_static_path = ["_static"]
html_theme_options = {
    "light_logo": "assets/Logo_white.png",
    "dark_logo": "assets/Logo_transparent.png",
}
favicons = ["assets/Logo_black.ico"]
