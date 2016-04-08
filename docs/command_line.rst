Command line usage
==================
The :mod:`psyplot.main` module defines a simple parser to parse commands
from the command line to make a plot of data in a netCDF file. Note that the
arguments change slightly if you have the ``psyplot_gui`` module installed
(see psyplot_gui_ documentation)

.. highlight:: bash

.. argparse::
   :module: psyplot.main
   :func: get_parser
   :prog: psyplot

.. _psyplot_gui: http://psyplot_gui.readthedocs.org/en/latest/command_line.html

