.. _getting-started:

Getting startet
===============

.. include:: <isonum.txt>

Initialization and interactive usage
------------------------------------

This section shall introduce you how to read data from a netCDF file and
visualize it via psyplot. For this, you need to have netCDF4_ and cartopy_ to
be installed (see
:ref:`install`).

.. _netCDF4: https://github.com/Unidata/netcdf4-python
.. _cartopy: http://scitools.org.uk/cartopy

After you :ref:`installed psyplot <install>`, you can import the package via

.. ipython::

    In [1]: import psyplot

Psyplot has several modules and subpackages. The main module for the use of
psyplot is the :mod:`~psyplot.project` module.

.. ipython::

    In [2]: import psyplot.project as psy

To plot a 2-dimensional variable (say temperature ``'t2m'`` from a netCDF file (say
``'demo.nc'``) type

.. ipython::

    @savefig docs_getting_started.png width=4in
    In [3]: maps = psy.plot.mapplot('demo.nc', name='t2m')

    @suppress
    In [4]: maps.close(True, True)

Instead of the filename ``'demo.nc'`` you can also insert a
:class:`xarray.Dataset`. Just as an example, you can use the
:func:`~psyplot.data.open_dataset` function via

.. ipython::

    In [4]: ds = psy.open_dataset('demo.nc')

    In [5]: ds.t2m

    In [6]: maps = psy.plot.mapplot(ds, name='t2m')

Psyplot allows you an interactive configuration of your formatoptions (see
:ref:`intro_fmt`) and the dimensions you are visualizing (see
:ref:`intro_dims`). For this purpose, you can use the :meth:`~ArrayList.update`
method of your project. For example: to update to the second timestep and the
title of your plot, type

.. ipython::

    @savefig docs_getting_started_update.png width=4in
    In [7]: maps.update(time=1, title='my title')

    @suppress
    In [8]: maps.close(True, True)

.. note::
    By default, a call of this method forces an automatical update and
    redrawing of all the plots. However you can modify this using the
    :attr:`~psyplot.data.ArrayList.no_auto_update` attribute (see
    :ref:`intro_auto_updates`).

It is recommended to close the plots after your work is done to delete the
data out of your memory. You can do this via the
:meth:`~psyplot.project.Project.close` method

.. ipython::

    In [7]: maps.close(True, True)

For other plotting methods of the :attr:`~psyplot.project.plot` attribute, see
the :class:`psyplot.plotter.ProjectPlotter` class.


.. _intro_dims:

Choosing the dimension
----------------------

As you see above, the scalar variable ``'t2m'`` has multiple time steps
(dimension ``'time'``) and multiple vertical levels (dimensions ``'lev'``). By
default, the :meth:`~psyplot.project.ProjectPlotter.mapplot` chooses the first
time step and the first vertical level (if those dimensions exist).

However, you can also specify the exact data slice for your visualization based
upon the dimensions in you dataset. When doing that, you basically do not have
to care about the exact dimension names in the netCDF files, because those are
encoded following the `CF Conventions <http://cfconventions.org/>`__. Hence
each of the above dimensions are assigned to one of the general dimensions
``'t'`` (time), ``'z'`` (vertical dimension), ``'y'`` (horizontal North-South
dimension) and ``'x'`` (horizontal East-West dimension). Hence, the dimensions
in ``'demo.nc'`` are encoded via ``'time'`` |rarr| ``'t'``, ``'lev'`` |rarr|
``'z'``, ``'lon'`` |rarr| ``'x'``, ``'lat'`` |rarr| ``'y'``.

Hence it is equivalent if you type

.. ipython::

    In [7]: maps = psy.plot.mapplot('demo.nc', name='t2m', t=1)

    @suppress
    In [8]: maps.close(True, True)

or

.. ipython::

    In [8]: maps = psy.plot.mapplot('demo.nc', name='t2m', time=1)

    @suppress
    In [9]: maps.close(True, True)

and finally you can also be very specific using the `dims` keyword via

.. ipython::

    In [9]: maps = psy.plot.mapplot('demo.nc', name='t2m', dims={'time': 1})

    @suppress
    In [10]: maps.close(True, True)

You can also use the `method` keyword from the plotting function to use the
advantages of the :meth:`xarray.DataArray.sel` method. E.g. to plot the data
corresponding to March 1979 you can use

.. ipython::

    In [10]: maps = psy.plot.mapplot('demo.nc', name='t2m', t='1979-03', method='nearest')

    @suppress
    In [11]: maps.close(True, True)

If you chose the wrong dimension in the beginning or want to change it during
the analization, it is best to use the :meth:`~ArrayList.update` method via

.. ipython::

    In [11]: maps.update(time=0)

    # or
    In [12]: maps.update(time='1979-01')


.. note::

    If your netCDF file does (for whatever reason) not follow the CF Conventions,
    we interprete the last dimension as the *x*-dimension, the second
    last dimension (if existent) as the *y*-dimension, the third last dimension as
    the *z*-dimension and the very first dimension as the *t*-dimension. If that
    still does not fit your netCDF files, you can specify the corresponding keys in
    the configuration dictionary :attr:`~psyplot.config.rcsetup.rcParams`, namely

    .. ipython::

        In [11]: psy.rcParams.find_all('decoder.(x|y|z|t)')


.. _intro_fmt:

Configuring the appearance of the plot
--------------------------------------

Since psyplot is build upon the great functionality of the matplotlib package,
you have the full control over the appearance of your plot. The *grassroots
methodology* would be to access the axes itself and modify it with the methods
of matplotlibs :class:`~matplotlib.axes.Axes` class.

However each plotting method of the :class:`~psyplot.project.ProjectPlotter`
class (i.e. each method of :attr:`psyplot.project.plot`) has several so-called
*formatoptions* that configure the appearance of the plot. Those formatoptions
are all designed for an interactive usage and can usually be controlled with
very simple commands. They range from simple formatoptions like :attr:`choosing
the title <psyplot.plotter.maps.FieldPlotter.title>` to formatting the
:attr:`boundaries of the colorbar <psyplot.plotter.maps.FieldPlotter.bounds>`
and colorbar. The formatoptions depend on the specific plotting method and
can be seen via the methods

.. autosummary::
    :toctree: generated/

    ~psyplot.project._PlotterInterface.keys
    ~psyplot.project._PlotterInterface.summaries
    ~psyplot.project._PlotterInterface.docs

For example to look at the formatoptions of the
:attr:`~psyplot.project.ProjectPlotter.mapplot` method in an interactive
session, type

.. ipython::

    In [24]: psy.plot.mapplot.keys(grouped=True)  # to see the fmt keys

    In [25]: psy.plot.mapplot.summaries(['title', 'cbar'])  # to see the fmt summaries

    In [26]: psy.plot.mapplot.docs('title')  # to see the full fmt docs

But of course you can also use the
:class:`online documentation <psyplot.project.ProjectPlotter>` of the
method your interested in.

To include a formatoption from the beginning, you can simply pass in the key
and the desired value as keyword argument, e.g.

.. ipython::

    In [27]: maps = psy.plot.mapplot('demo.nc', name='t2m', title='my title',
       ....:                         cbar='r')

    @suppress
    In [28]: maps.close(True, True)

This works generally well as long as there are no dimensions in the desired
data with the same name as one of the passed in formatoptions. If you want to
be really sure, use the `fmt` keyword via

.. ipython::

    In [28]: maps = psy.plot.mapplot('demo.nc', name='t2m', fmt={'title': 'my title',
       ....:                                              'cbar': 'r'})

    @suppress
    In [29]: maps.close(True, True)

The same methodology works for the interactive usage, i.e. you can use

.. ipython::

    In [29]: maps.update(title='my title', cbar='r')

    # or
    In [30]: maps.update(fmt={'title': 'my title', 'cbar': 'r'})


.. _intro_update:

Controlling the update
----------------------

.. _intro_auto_updates:

Automatic update
^^^^^^^^^^^^^^^^

By default, a call of this method forces an automatical update and redrawing
of all the plots. There are however several ways to modify this behaviour:

1. Changing the behaviour of one single project

   1. in the initialization of a project using the `auto_update` keyword

      .. ipython::

          In [9]: maps = psy.plot.mapplot('demo.nc', name='t2m', auto_update=False)

          @suppress
          In [10]: maps.close(True, True)

   2. setting the :attr:`~psyplot.data.ArrayList.no_auto_update` attribute

      .. ipython::

          In [10]: maps.no_auto_update = True

2. Changing the default configuration in the
   :attr:`~psyplot.rcsetup.config.rcParams` ``'lists.auto_update'`` item

   .. ipython::

        In [11]: psy.rcParams['lists.auto_update'] = False

        @suppress
        In [11]: psy.rcParams['lists.auto_update'] = True

3. Using the :attr:`~psyplot.data.ArrayList.no_auto_update` attribute as a
   context manager

    .. ipython::

        In [12]: with maps.no_auto_update:

           ....:    maps.update(title='test')

If you disabled the automatical update via one of the above methods, you have
to start the registered updates manually via

.. ipython::

    In [12]: maps.update(auto_update=True)

    # or
    In [13]: maps.start_update()

Direct control on formatoption update
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, when updating a formatoption, it is checked for each plot whether
the formatoption would change during the update or not. If not, the
formatoption is not updated. However, sometimes you may want to do that and
for this, you can use the `force` keyword or the
:meth:`~psyplot.data.ArrayList.update` method.


.. _intro_multiple:

Creating and managing multiple plots
------------------------------------

One major advantage of the psyplot project is the systematic manamgement of
multiple plots at the same time. To create multiple plots, simply pass in a
list of dimension values and/or names. For example

.. ipython::

    In [12]: maps = psy.plot.mapplot('demo.nc', name='t2m', time=[0, 1])

    In [13]: maps

    @suppress
    In [14]: maps.close(True, True)

Will create two plots: one for the first and one for the second time step.
Furthermore

.. ipython::

    In [14]: maps = psy.plot.mapplot('demo.nc', name=['t2m', 'u'], time=[0, 1])

    In [15]: maps

    @suppress
    In [16]: maps.close(True, True)

will created four plots. You can also use the `ax` keyword to setup how the
plots will be arranged (by default, each plot is made on in own figure) and
the `sort` keyword to specify how the plots will be sorted.

As an example we plot the variables ``'t2m'`` and ``'u'`` for the first and
second time step into one figure and sort by time. This will produce

.. ipython::

    @savefig docs_multiple_plots.png width=4in
    In [16]: maps = psy.plot.mapplot(
       ....:     'demo.nc', name=['t2m', 'u'], time=[0, 1], ax=(2, 2), sort=['time'],
       ....:     title='%(long_name)s, %b')

    In [17]: maps

    @suppress
    In [18]: maps.close(True, True)

.. warning::

    As the xarray package, the slicing is based upon positional indexing with
    lists (see `the xarray documentation on ositional indexing
    <http://xarray.pydata.org/en/stable/indexing.html#positional-indexing>`__).
    Hence you might think of choosing your data slice via
    ``psy.plot.mapplot(..., x=[1, 2, 3, 4, 5], ...)``. However this would result
    in 5 different plots! Instead you have to write
    ``psy.plot.mapplot(..., x=[[1, 2, 3, 4, 5]], ...)``. The same is true
    for plotting methods like the
    :attr:`~psyplot.project.ProjectPlotter.mapvector` method. Since this
    method needs two variables (one for the latitudinal and one for the
    longitudinal direction), typing

    .. ipython::
        :okexcept:

        In [18]: maps = psy.plot.mapvector('demo.nc', name=['u', 'v'])

    results in a :class:`ValueError`. Instead you have to write

    .. ipython::

        In [19]: maps = psy.plot.mapvector('demo.nc', name=[['u', 'v']])

        @suppress
        In [20]: maps.close(True, True)

    Please have a look into the documentations of the
    :attr:`~psyplot.project.ProjectPlotter.mapvector` and
    :attr:`~psyplot.project.ProjectPlotter.mapcombined` for getting examples
    on how to call this functions.

Finally, if you want to choose a specific array, you can use meta attributes,
dimensions and the specific :attr:`~psyplot.data.InteractiveBase.arr_name`
attribute and type

.. ipython

    @suppress
    In [16]: maps = psy.plot.mapplot(
       ....:     'demo.nc', name=['t2m', 'u'], time=[0, 1], ax=(2, 2), sort=['time'],
       ....:     title='%(long_name)s, %b')

    In [20]: maps(t=0)

    In [21]: maps(t='1979-01')

    In [22]: maps(name='t2m')

    In [23]: maps(long_name='Temperature')

This behaviour is especially useful if you want to address only some arrays
with your update. For example, let's consider we want to choose a ``'winter'``
colormap for the zonal wind variable and a colormap ranging from blue to red
for the temperature. Then we could do this via

.. ipython::

    In [24]: maps(name='t2m').update(cmap='RdBu_r')

    In [26]: maps(name='u').update(cmap='winter')

However, we recommend to temporarily disable the automatic update. This will
first register the updates and not draw the figures before we want them to.

Hence, it is better to use the context manager
:attr:`~psyplot.data.ArrayList.no_auto_update` (see :ref:`intro_auto_updates`)

.. ipython::

    In [27]: with maps.no_auto_update:
       ....:     maps(name='t2m').update(cmap='RdBu_r')
       ....:     maps(name='u').update(cmap='winter')
       ....:     maps.start_update()

    @suppress
    In [28]: maps.close(True, True)

Finally you can access the plots created by a specific plotting method
through the corresponding attribute in the :class:`~psyplot.project.Project`
class. In this case this is of course useless because all plots in ``maps``
were created by the same plotting method, but it may be helpful when having
different plotters in one project (see :ref:`framework`). However, the plots
created by the :attr:`~psyplot.project.ProjectPlotter.mapplot` method could be
accessed via

.. ipython::

    In [24]: maps.mapplot

Saving and loading your project
-------------------------------

The :class:`~psyplot.project.Project` class allows you to save the settings
of your plot and load them again to restore your plots or apply your settings
to different data.

To save your project, use the :meth:`~psyplot.project.Project.save_project`
method:

.. ipython::

    @suppress
    In [28]: maps = psy.plot.mapplot('demo.nc', name='t2m', time=[0, 1], ax=(1, 2))

    @verbatim
    In [28]: maps.save_project('test.pkl')

This saves the plot-settings into the file ``'test.pkl'``. In order to not get
to large project files, we do not store the data but only the filenames of the
datasets. Hence, if you want to load the project again, make sure that the
datasets are accessible through the path as they are listed in the
:attr:`~psyplot.project.Project.dsnames` attribute.

Otherwise you have several options to avoid wrong paths:

1. Use the `alternative_paths` parameter and provide for each filename a
   specific path

   .. ipython::

       In [29]: maps.dsnames

       @verbatim
       In [30]: maps.save_project(
          ....:     'test.pkl', alternative_paths={'demo.nc': 'other_path.nc'})

2. pack the whole data to the place where you store the project file

   .. ipython::

       @verbatim
       In [31]: maps.save_project('test.pkl', pack=True)

3. specify where the datasets can be found when you load the data:

   .. ipython::

       @verbatim
       In [32]: maps = psy.Project.load_project(
          ....:     'test.pkl', alternative_paths={'demo.nc': 'other_path.nc'})

       @suppress
       In [32]: maps.close(True, True)

To restore your project, simply use the
:meth:`~psyplot.project.Project.load_project` method via

.. ipython::

    @verbatim
    In [33]: maps = psy.Project.load_project('test.pkl')

.. note::

    Saving a project stores the figure informations like axes positions,
    background colors, etc. However only the axes informations from the data
    objects managed by this project are saved but no other axes. For example
    saving a plot with some additional axes

    .. ipython::

        In [33]: fig, axes = plt.subplots(1, 2)

        In [34]: lines = psy.plot.lineplot('demo.nc', name=['t2m'], t=0, z=0, y=0,
           ....:                           ax=axes[0])

        @savefig docs_getting_started_saved.png width=4in
        In [35]: axes[1].plot([6, 7])

        @verbatim
        In [36]: settings = lines.save_project()

        @suppress
        In [36]: settings = lines.save_project(use_rel_paths=False)

    and loading the project again

    .. ipython::

        In [37]: lines.close(True, True)

        @savefig docs_getting_started_loaded.png width=4in
        In [38]: lines = psy.Project.load_project(settings)

        @suppress
        In [39]: lines.close(True, True)

    restores only the first axes, but not the second. To fix this, you could
    create the figure configuration by yourself and use the `alternative_axes`
    parameter when loading the project

    .. ipython::

        In [39]: fig, axes = plt.subplots(1, 2)

        In [40]: lines = psy.Project.load_project(
           ....:     settings, alternative_axes=[axes[0]])

        @savefig docs_getting_started_restored.png width=4in
        In [41]: axes[1].plot([6, 7])
