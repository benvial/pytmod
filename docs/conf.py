# Authors: Benjamin Vial
# This file is part of pytmod
# License: GPLv3
# See the documentation at bvial.info/pytmod


from __future__ import annotations

import datetime
import json
import re
import shutil
import subprocess
import warnings
from pathlib import Path

from packaging.version import Version
from sphinx.application import Sphinx
from sphinx.util import logging

import pytmod as package

logger = logging.getLogger(__name__)


def get_all_version_tags():
    """Get all version tags from git, sorted by version number (newest first)."""
    try:
        result = subprocess.run(
            ["git", "tag"], capture_output=True, text=True, check=True
        )
        tags = result.stdout.splitlines()
        version_tags = [tag for tag in tags if re.fullmatch(r"v\d+\.\d+\.\d+", tag)]
        if not version_tags:
            return []
        return sorted(version_tags, key=lambda v: Version(v[1:]), reverse=True)
    except subprocess.CalledProcessError:
        return []


def get_latest_version_tag():
    """Get the latest version tag (highest version number)."""
    versions = get_all_version_tags()
    return versions[0] if versions else None


def get_current_version():
    """Get the current version from the package."""
    return package.__version__


latest_tag = get_latest_version_tag()
current_version = get_current_version()

redirect_contents = """
<!DOCTYPE html>
<html>
  <head>
    <title>Redirecting to latest version</title>
    <meta charset="utf-8" />
    <meta http-equiv="refresh" content="0; url=./latest/index.html" />
    <link rel="canonical" href="https://benvial.github.io/pytmod/latest/index.html" />
  </head>
</html>
"""


def build_multiversion_docs(
    app: Sphinx,
) -> None:
    """
    Build documentation for multiple versions.
    This function is called after the main build to create versioned docs.
    """
    if not app.config.versions:
        return

    outdir = Path(app.outdir).parents[0]

    # Get all version tags
    version_tags = get_all_version_tags()
    if not version_tags:
        logger.warning("No version tags found for multiversion build")
        return

    # Get the latest tag
    latest = version_tags[0]

    # Create versions.json for templates
    versions_data = []
    for tag in version_tags:
        version_name = tag[1:]  # Remove 'v' prefix
        versions_data.append(
            {
                "name": version_name,
                "url": f"./{tag}/index.html",
                "is_latest": tag == latest,
            }
        )

    # Write versions.json
    versions_json_path = outdir / "versions.json"
    with Path.open(versions_json_path, "w") as f:
        json.dump(versions_data, f)

    logger.info(f"Built {len(version_tags)} versions: {version_tags}")
    logger.info(f"Latest version: {latest}")

    # Rename latest tag directory to 'latest'
    try:
        latest_dir = Path(outdir) / latest
        latest_dir = latest_dir.with_name(latest)
        if latest_dir.exists():
            new_dir = Path(outdir) / "latest"
            if new_dir.exists():
                shutil.rmtree(new_dir)
            shutil.move(str(latest_dir), str(new_dir))
            logger.info(f"Renamed {latest} to latest")
    except Exception as e:
        logger.warning(f"Could not rename latest directory: {e}")

    # Rename main tag directory to 'dev'

    logger.info("Renaming main to dev")
    try:
        main_dir = Path(outdir) / "main"
        main_dir = main_dir.with_name("main")
        if main_dir.exists():
            new_dir = Path(outdir) / "dev"
            if new_dir.exists():
                shutil.rmtree(new_dir)
            shutil.move(str(main_dir), str(new_dir))
            logger.info("Renamed main to dev")
    except Exception as e:
        logger.warning(f"Could not rename main directory: {e}")

    # Write redirect index.html
    index_path = outdir / "index.html"
    with Path.open(index_path, "w") as f:
        f.write(redirect_contents)
    logger.info("Wrote redirect index.html")


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# see https://shibuya.lepture.com/extensions/sphinx-copybutton/
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    "autoapi.extension",
    "myst_parser",
    "sphinxcontrib.bibtex",
    "sphinx_design",
    "sphinx_tabs.tabs",
    "sphinx_togglebutton",
    "numpydoc",
    "sphinx_iconify",
]

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]


bibtex_bibfiles = ["_static/biblio.bib"]
bibtex_default_style = "unsrt"
# bibtex_reference_style = "label"


# -- autoapi configuration ---------------------------------------------------

autodoc_typehints = "signature"  # autoapi respects this

autoapi_type = "python"
autoapi_dirs = ["../pytmod"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_add_toctree_entry = False
# autoapi_python_use_implicit_namespaces = True
autoapi_keep_files = True
# autoapi_generate_api_docs = False


def skip_member(app, what, name, obj, skip, options):  # noqa: ARG001
    # skip submodules
    if what == "module":
        skip = True
    return skip


versions = get_all_version_tags()
versions_tuple = [("latest", "/")] + [(v, f"/{v}/") for v in versions[:-1]]
html_context = {
    "versions": versions_tuple,
    "current_version": "latest",
}


def update_version_context(app, pagename, templatename, context, doctree):  # noqa: ARG001
    """Update template context with version information."""
    if app.config.versions:
        versions = get_all_version_tags()
        latest = versions[0] if versions else None

        context["versions"] = []
        for tag in versions:
            version_name = tag[1:]  # Remove 'v' prefix
            context["versions"].append(
                {
                    "name": version_name,
                    "url": f"./{tag}/index.html",
                    "is_latest": tag == latest,
                }
            )

        context["current_version"] = {
            "name": current_version,
            "url": f"./{current_version}/index.html",
            "is_latest": current_version == (latest[1:] if latest else None),
        }

        context["latest_version"] = {
            "name": latest[1:] if latest else None,
            "url": "./latest/index.html",
            "is_latest": True,
        }


def setup(app):
    app.connect("autoapi-skip-member", skip_member)
    app.add_config_value("versions", False, "env")  # Default is False
    app.add_config_value("current_version", current_version, "env")
    app.add_config_value(
        "latest_version", latest_tag[1:] if latest_tag else None, "env"
    )

    versions = app.config.versions
    if versions:
        logger.info("Building multiple versions of docs")
        app.connect("build-finished", build_multiversion_docs)
        app.connect("html-page-context", update_version_context)
    else:
        # Single version build - just rename to latest
        app.connect("build-finished", build_multiversion_docs)


conf_dir = Path(__file__).parent.resolve()
static = conf_dir / "_static"


# Add any paths that contain templates here, relative to this directory.
templates_path = [str(conf_dir / "_templates")]

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "pytmod"

copyright = f"Copyright © {datetime.date.today().year}, Benjamin Vial"

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
html_theme_options = {
    "github_url": "https://github.com/benvial/pytmod",
    "light_logo": "_static/pytmod-name.svg",
    "dark_logo": "_static/pytmod-name-dark.svg",
    "accent_color": "blue",
    # "announcement": "This is a community project. Any contribution is welcome!",
}
# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = []

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "pytmod"

# A shorter title for the navigation bar.  Default is the same as html_title.
# html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/pytmod-name.svg"  # str(static / "pytmod-name.svg")

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/favicon.ico"  # str(static / "favicon.ico")

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["version-banner.js"]


# # Custom sidebar templates, maps document names to template names.
# html_sidebars = {"**": ["sidebar_links.html"]}

# html_sidebars = [
#     "banner-version.html",
# ]

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
    # "ignore_pattern": r"^(?!plot_).*",  # ignore files that do not start with plot_
    # "ignore_pattern": r"^((?!/plot_).)*$",  # ignore files that do not start with plot_
    # "first_notebook_cell": (
    #     "import matplotlib\n" "mpl.style.use('gyptis')\n" "%matplotlib inline"
    # ),
    # "image_scrapers": ("matplotlib", PNGScraper()),
    # Modules for which function level galleries are created.
    "doc_module": package.__name__,
    "thumbnail_size": (800, 800),
    "default_thumb_file": str(static / "pytmod.png"),
    "show_memory": True,
    "matplotlib_animations": (True, "html5"),
    # "binder": {
    #     "org": "your-org",
    #     "repo": "your-org.gitlab.io/repo",
    #     "branch": "doc",
    #     "binderhub_url": "https://mybinder.org",
    #     "dependencies": "../environment.yml",
    #     "notebooks_dir": "notebooks",
    #     "use_jupyter_lab": True,
    # },
}

# Remove matplotlib agg warnings from generated doc when using plt.show
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a"
    " non-GUI backend, so cannot show the figure.",
)
