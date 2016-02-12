.. _install:

Installation
============

How to install
--------------

Installation using conda
^^^^^^^^^^^^^^^^^^^^^^^^
We highly recommend to use conda_ for installing psyplot. After downloading
the installer from anaconda_, you can install psyplot simply via::

    $ conda install -c chilipp psyplot

If you want to use psyplot for visualizing geo-referenced data on a map, you
furthermore have to install cartopy_ via::

    $ conda install -c scitools cartopy

If you want to be able to read and write netCDF files, you can use for example
the netCDF4_ package via::

    $ conda install netCDF4

If you want to be able to read GeoTiff Raster files, you will need to have
gdal_ installed::

    $ conda install gdal

Please also visit the `xarray installation notes`_
for more informations on how to best configure the `xarray`_
package for your needs.

Installation using pip
^^^^^^^^^^^^^^^^^^^^^^
If you do not want to use conda for managing your python packages, you can also
use the python package manager ``pip`` and install via::

    $ pip install psyplot


Dependencies
------------
Required dependencies
^^^^^^^^^^^^^^^^^^^^^
Psyplot has been tested for python 2.7 and 3.4. Furthermore the package is
built upon multiple other packages, namely

- xarray_>=0.7: Is used for the data management in the psyplot package
- matplotlib_>=1.4.3: **The** python visualiation
  package
- `PyYAML <http://pyyaml.org/>`__: Needed for the configuration of psyplot

Optional dependencies
^^^^^^^^^^^^^^^^^^^^^
We furthermore recommend to use

- cartopy_: For visualizing Georeferenced data using the plotters from the
  :mod:`psyplot.plotter.maps` module.
- seaborn_: For modifying the default style of matplotlib figures and making
    :attr:`violinplots <psyplot.project.ProjectPlotter.violinplot`
- dask_: For an efficient management of the data
- netCDF4_: For reading and writing netCDF files
- gdal_: For reading GeoTiff Rasters using the
  :class:`psyplot.gdal_store.GdalStore` or the
  :func:`~psyplot.data.open_dataset` function.
- `ipython <http://ipython.org/>`__: For using psyplot interactively

.. _conda: http://conda.io/
.. _anaconda: https://www.continuum.io/downloads
.. _cartopy: http://scitools.org.uk/cartopy/docs/latest/installing.html
.. _seaborn: http://stanford.edu/~mwaskom/software/seaborn/
.. _netCDF4: https://github.com/Unidata/netcdf4-python
.. _matplotlib: http://matplotlib.org
.. _gdal: http://www.gdal.org/
.. _dask: http://dask.pydata.org
.. _xarray installation notes: http://xarray.pydata.org/en/stable/installing.html
.. _xarray: http://xarray.pydata.org/

Running the tests
-----------------
Check out the github_ repository and navigate to the ``'tests'`` directory.
You can either simply run::

    $ python main.py

or install the pytest_ module and run::

    $ py.test

or in the main directory::

    $ python setup.py pytest

Building the docs
-----------------
To build the docs, check out the github_ repository and install the
requirements in ``'docs/environment.yml'``. The easiest way to do this is via
anaconda by typing::

    $ conda env create -n psyplot -f docs/environment.yml
    $ source activate psyplot_docs

Then build the docs via::

    $ cd docs
    $ make html

.. _github: https://github.com/Chilipp/psyplot
.. _pytest: https://pytest.org/latest/contents.html
