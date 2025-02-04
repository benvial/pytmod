#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod

import os
import shutil
import subprocess
import sys
from os.path import join, splitext

import datetime
from dataclasses import asdict
import pytmod as package
from pybtex.plugin import register_plugin
from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from pybtex.style.labels import BaseLabelStyle


# # finished generated, so go back up a level
# os.chdir("..")

# # If extensions (or modules to document with autodoc) are in another directory,
# # add these directories to sys.path here. If the directory is relative to the
# # documentation root, use os.path.abspath to make it absolute, like shown here.
# # sys.path.insert(0, os.path.abspath(join('..', 'pytmod')))
# sys.path.insert(0, os.path.abspath(".."))

# print sys.path

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
    "autoapi.extension",
    "myst_parser",
    "sphinxcontrib.bibtex",
]


# register_plugin("pybtex.style.formatting", "custombibstyle", CustomBibStyle)

bibtex_bibfiles = ["_static/biblio.bib"]
bibtex_default_style = "unsrt"
# bibtex_reference_style = "label"


autoapi_dirs = ["../pytmod"]
autoapi_options = ["members", "undoc-members"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "pytmod"

copyright = f"{datetime.date.today().year}, Benjamin Vial"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.


version = package.__version__
# The full version, including alpha/beta/rc tags.
release = version

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
# language = None

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
# default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "shibuya"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# html_theme_options = {}

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "pytmod"

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/pytmod.svg"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]


# # Custom sidebar templates, maps document names to template names.
# html_sidebars = {"**": ["sidebar_links.html"]}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
# html_domain_indices = True

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
# html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
# html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
# html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = "pytmoddoc"


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    ("index", "pytmod.tex", "pytmod Documentation", "David Powell", "manual"),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [("index", "pytmod", "pytmod Documentation", ["Benjamin Vial"], 1)]

# If true, show URL addresses after external links.
# man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        "index",
        "pytmod",
        "pytmod Documentation",
        "Benjamin Vial",
        "pytmod",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# Documents to append as an appendix to all manuals.
# texinfo_appendices = []

# If false, no module index is generated.
# texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
# texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
# texinfo_no_detailmenu = False

sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": ["../examples"],
    # path where to save gallery generated examples
    "gallery_dirs": ["examples"],
    # # path to your examples scripts
    # "examples_dirs": ["../tutorials"],
    # # path where to save gallery generated examples
    # "gallery_dirs": ["tutorials"],
    # directory where function granular galleries are stored
    "backreferences_dir": "generated/backreferences",
    "remove_config_comments": True,
    "reference_url": {
        "sphinx_gallery": None,
    },
    "reset_modules": (),
    "image_scrapers": ("matplotlib"),
    # "pypandoc": True,
    # "pypandoc": {"extra_args": ["-C","--bibliography=_custom/latex/biblio.bib"], "filters": []},
    # "filename_pattern": "plot_homogenization\.py",
    "filename_pattern": "/plot_",
    "ignore_pattern": r"^(?!plot_).*",  # ignore files that do not start with plot_
    # "ignore_pattern": r"^((?!/plot_).)*$",  # ignore files that do not start with plot_
    # "first_notebook_cell": (
    #     "import matplotlib\n" "mpl.style.use('gyptis')\n" "%matplotlib inline"
    # ),
    # "image_scrapers": ("matplotlib", PNGScraper()),
    # Modules for which function level galleries are created.
    "doc_module": package.__name__,
    "thumbnail_size": (800, 800),
    "default_thumb_file": "./_static/pytmod.png",
    "show_memory": True,
    # "binder": {
    #     "org": "phokaia",
    #     "repo": "phokaia.gitlab.io/emustack",
    #     "branch": "doc",
    #     "binderhub_url": "https://mybinder.org",
    #     "dependencies": "../environment.yml",
    #     "notebooks_dir": "notebooks",
    #     "use_jupyter_lab": True,
    # },
}


html_theme_options = {"github_url": "https://github.com/benvial/pytmod"}
