# Configuration file for the Sphinx documentation builder.
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "PyGlobalSearch"
copyright = "2025, Germán Martín Heim"
author = "Germán Martín Heim"
release = "0.4.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_design",
]

# Autosummary settings
autosummary_generate = True
autosummary_generate_overwrite = True

# Autodoc settings
autoclass_content = "both"
autodoc_typehints = "none"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_nav_level": 3,
    "navigation_depth": 3,
    "collapse_navigation": False,
    "show_toc_level": 2,
    "logo": {
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
    },
    "icon_links": [
        {
            "name": "Website",
            "url": "https://germanheim.github.io/globalsearch-rs-website/",
            "icon": "fas fa-globe",
        },
        {
            "name": "GitHub",
            "url": "https://github.com/GermanHeim/globalsearch-rs",
            "icon": "fab fa-github-square",
        },
    ],
}

html_static_path = ["_static"]

# Add custom CSS for logo spacing
html_css_files = [
    "custom.css",
]
