# -*- coding: utf-8 -*-
#
# psyplot documentation build configuration file, created by
# sphinx-quickstart on Mon Jul 20 18:01:33 2015.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

import sphinx
import os
import os.path as osp
import sys
import re
import six
import subprocess as spr
from itertools import product
import warnings
# make sure, psyplot from parent directory is used
sys.path.insert(0, os.path.abspath('..'))
import psyplot
from psyplot.plotter import Formatoption, Plotter

# automatically import all plotter classes
psyplot.rcParams['project.auto_import'] = True
# include links to the formatoptions in the documentation of the
# :attr:`psyplot.project.ProjectPlotter` methods
Plotter.include_links(True)

warnings.filterwarnings('ignore', message="axes.color_cycle is deprecated")
warnings.filterwarnings(
    'ignore', message=("This has been deprecated in mpl 1.5,"))
warnings.filterwarnings('ignore', message="invalid value encountered in ")

# -- General configuration ------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'matplotlib.sphinxext.plot_directive',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'sphinxarg.ext',
    'psyplot.sphinxext.extended_napoleon',
    'autodocsumm',
    'sphinx_nbexamples',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# on_rtd is whether we are on readthedocs.org, this line of code grabbed from
# docs.readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

# process the examples if they don't exist already
process_examples = not osp.exists(osp.join(osp.dirname(__file__), 'examples'))

if on_rtd:
    spr.call([sys.executable] +
             ('-m ipykernel install --user --name python3 '
              '--display-name python3').split())

# create the api documentation
if not osp.exists(osp.join(osp.dirname(__file__), 'api')):
    spr.check_call(['bash', 'apigen.bash'])

# The cdo example would require the installation of climate data operators
# which is a bit of an overkill
example_gallery_config = dict(
    dont_preprocess=['../examples/example_cdo.ipynb'],
    urls='https://github.com/Chilipp/psyplot/blob/v1.0.0.dev0/examples',
    )

napoleon_use_admonition_for_examples = True

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

autodoc_default_flags = ['show_inheritance', 'autosummary']
autoclass_content = 'both'

not_document_data = ['psyplot.config.rcsetup.defaultParams',
                     'psyplot.config.rcsetup.rcParams']

ipython_savefig_dir = os.path.join(os.path.dirname(__file__), '_static')

# General information about the project.
project = u'psyplot'
copyright = u'2015, Philipp Sommer'
author = u'Philipp Sommer'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = re.match('\d+\.\d+\.\d+', psyplot.__version__).group()
# The full version, including alpha/beta/rc tags.
release = psyplot.__version__

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

    # Add any paths that contain custom static files (such as style sheets)
    # here, relative to this directory. They are copied after the builtin
    # static files, so a file named "default.css" will overwrite the builtin
    # "default.css".
    html_static_path = ['_static']

# otherwise, readthedocs.org uses their theme by default, so no need to specify

# Output file base name for HTML help builder.
htmlhelp_basename = 'psyplotdoc'

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = '_static/psyplot.png'

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = '_static/psyplot.ico'

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    'index': ['sidebarlogo.html', 'sidebarusefullinks.html', 'searchbox.html'],
    '**': ['sidebarlogo.html', 'relations.html', 'searchbox.html', 
           'localtoc.html', 'sidebarusefullinks.html']
}

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # Additional stuff for the LaTeX preamble.
    'preamble': '\setcounter{tocdepth}{10}'
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
  (master_doc, 'psyplot.tex', u'psyplot Documentation',
   u'Philipp Sommer', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'psyplot', u'psyplot Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
  (master_doc, 'psyplot', u'psyplot Documentation',
   author, 'psyplot', 'Python framework for interactive data documentation',
   'Miscellaneous'),
]


# -- Options for Epub output ----------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'pandas': ('http://pandas.pydata.org/pandas-docs/stable/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'matplotlib': ('http://matplotlib.org/', None),
    'sphinx': ('http://www.sphinx-doc.org/en/stable/', None),
    'xarray': ('http://xarray.pydata.org/en/stable/', None),
    'cartopy': ('http://scitools.org.uk/cartopy/docs/latest/', None),
    'mpl_toolkits': ('http://matplotlib.org/basemap/', None),
    'psy_maps': (
        'https://psyplot.readthedocs.io/projects/psy-maps/en/latest/', None),
    'psy_simple': (
        'https://psyplot.readthedocs.io/projects/psy-simple/en/latest/', None),
    'psy_reg': ('https://psyplot.readthedocs.io/projects/psy-reg/en/latest/',
                None),
    'psyplot_gui': (
        'http://psyplot.readthedocs.io/projects/psyplot-gui/en/latest/'
    )
}
if six.PY3:
    intersphinx_mapping['python'] = ('https://docs.python.org/3.6/', None)
else:
    intersphinx_mapping['python'] = ('https://docs.python.org/2.7/', None)


replacements = {
    '`psyplot.rcParams`': '`~psyplot.config.rcsetup.rcParams`',
    '`psyplot.InteractiveList`': '`~psyplot.data.InteractiveList`',
    '`psyplot.InteractiveArray`': '`~psyplot.data.InteractiveArray`',
    '`psyplot.open_dataset`': '`~psyplot.data.open_dataset`',
    '`psyplot.open_mfdataset`': '`~psyplot.data.open_mfdataset`',
    }


def link_aliases(app, what, name, obj, options, lines):
    for (key, val), (i, line) in product(six.iteritems(replacements),
                                         enumerate(lines)):
        lines[i] = line.replace(key, val)


fmt_attrs_map = {
    'Interface to other formatoptions': [
        'children', 'dependencies', 'connections', 'parents',
        'shared', 'shared_by'],
    'Formatoption intrinsic': [
        'value', 'value2share', 'value2pickle', 'default', 'validate'],
    'Interface for the plotter': [
        'lock', 'diff', 'set_value', 'check_and_set', 'initialize_plot',
        'update', 'share', 'finish_update', 'remove', 'changed', 'plotter',
        'priority', 'key', 'plot_fmt', 'update_after_plot',
        'requires_clearing'],
    'Interface to the data': ['data_dependent', 'index_in_list', 'project',
                              'ax', 'raw_data', 'decoder', 'any_decoder',
                              'data', 'iter_data', 'iter_raw_data',
                              'set_data', 'set_decoder'],
    'Information attributes': ['group', 'name', 'groupname', 'default_key'],
    'Miscellaneous': ['init_kwargs', 'logger'],
}


def group_fmt_attributes(app, what, name, obj, section, parent):
    if parent is Formatoption:
        return next(
            (group for group, val in fmt_attrs_map.items() if name in val),
            None)


def setup(app):
    app.connect('autodoc-process-docstring', link_aliases)
    app.connect('autodocsumm-grouper', group_fmt_attributes)
    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}
