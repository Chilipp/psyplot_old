.. _framework:

The psyplot framework
=====================
The main module we used so far, was the :mod:`psyplot.project` module. It is
the end of a whole framework that is setup by the psyplot package.

The full framework however is designed in analogy to matplotlibs
`figure - axes - artist setup <http://matplotlib.org/1.5.1/users/artists.html>`__,
where one figure controls multiple axis, an axes is the manager of multiple
artists (e.g. a simple line) and each artist is responsible for visualizing an
object on the plot. The psyplot framework instead is defined through the
:class:`~psyplot.project.Project` - (:class:`~psyplot.data.InteractiveBase` -
:class:`~psyplot.plotter.Plotter`) - :class:`~psyplot.plotter.Formatoption`
relationship.

The :func:`~psyplot.project.project`
------------------------------------

.. currentmodule:: psyplot.project

The :class:`psyplot.project.Project` class (in analogy to matplotlibs
:class:`~matplotlib.figure.Figure` class) is basicly a list controlling
multiple plot objects. It comprises the full functionality of the package and
packs it into one class, the :class:`~psyplot.project.Project` class.

In analogy to pyplots :func:`~matplotlib.pyplot.figure` function, a new project
can simply be created via

.. ipython::

    In [1]: import psyplot.project as psy

    In [2]: p = psy.project()

This automatically sets ``p`` to be the current project which can be accessed
through the :func:`gcp` method. You can also set the current
project by using the :func:`scp` function.

.. note::

    We highly recommend to use the :func:`project` function to create new
    projects instead of creating projects from the :class:`Project`. This
    ensures the right numbering of the projects of old projects

The project uses the plotters from the :mod:`psyplot.plotter` module to
visualize your data. Hence you can add new plots and new data to the project by
using the :attr:`Project.plot` attribute or the :attr:`psyplot.project.plot`
attribute which targets the current project. The return types of the plotting
methods are again instances of the :class:`Project` class, however we consider
them as *subprojects* in contrast *main projects*. There is basically no
difference but the result of the :attr:`Project.is_main` attribute which is
``False`` for subprojects. Hence, each new plot creates a subproject but also
stores the data array in the corresponding main project of the :class:`Project`
instance from which the plot method has been called. The newly created
subproject can be accessed via

.. ipython::

    In [3]: sp = psy.gcp()

whereas the current main project can be accessed via

.. ipython::

    In [4]: p = psy.gcp(main=True)

If there is no current subproject, the current main project is returned.

Plots created by a specific method of the :attr:`Project.plot` attribute may
however be accessed via the corresponding attribute of the :class:`Project`
class. The following example creates three subprojects, two with the
:attr:`~ProjectPlotter.mapplot` method and one with the simple
:attr:`~ProjectPlotter.lineplot` method to visualize simple lines.


.. ipython::

    In [5]: import matplotlib.pyplot as plt

    In [6]: import cartopy.crs as ccrs

    In [7]: ax = list(psy.multiple_subplots(2, 2, n=3, subplot_kw=dict(
       ...:                                 projection=ccrs.PlateCarree())))

    In [8]: ax.append(plt.gcf().add_subplot(2, 2, 4))

    # scalar field of the zonal wind velocity in the file demo.nc
    In [5]: psy.plot.mapplot('demo.nc', name='u', ax=ax[0])

    # a second scalar field of temperature
    In [6]: psy.plot.mapplot('demo.nc', name='t2m', time=1, ax=ax[1])

    # a vector plot projected on the earth
    In [7]: psy.plot.mapvector('demo.nc', name=[['u', 'v']], ax=ax[2],
       ...:                    attrs={'long_name': 'Wind speed'})

    @savefig docs_framework_project_demo1.png width=4in
    In [8]: psy.plot.lineplot('demo.nc', name='t2m', x=0, y=0, z=range(4),
       ...:                  ax=ax[3], xticklabels='%b %d')

The latter is now the current subproject we could access via ``psy.gcp()``.
However we can access all of the through the main project

.. ipython::

    In [9]: p = psy.gcp(True)

    In [10]: p  # all arrays

    In [11]: p.mapplot  # all scalar fields

    In [12]: p.mapvector  # all vector plots

    In [13]: p.maps  # all data arrays that are plotted on a map

    In [14]: p.lineplot # the simple plot we created

The advantage is, since every plotter has different formatoptions, we can
now update them very easily. For example lets update the arrowsize to
1 (which only works for the :attr:`~Project.mapvector` plots), the projection
to an orthogonal (which only works for :attr:`~Project.maps`), the simple
plots to use the ``'viridis'`` colormap for coding the lines and for all
we choose their title corresponding to the variable names

.. ipython::

    @suppress
    In [15]: with p.maps.no_auto_update:
       ....:     p.maps.update(grid_labels=False)

    In [15]: p.maps.update(projection='ortho')

    In [16]: p.mapvector.update(color='r', plot='stream', lonlatbox='Europe')

    In [17]: p.lineplot.update(color='coolwarm')

    @savefig docs_framework_project_demo2.png width=4in
    In [18]: p.update(title='%(long_name)s')



The :class:`~psyplot.data.InteractiveBase` and the :class:`~psyplot.plotter.Plotter` classes
--------------------------------------------------------------------------------------------

.. currentmodule:: psyplot.plotter

Interactive data objects
^^^^^^^^^^^^^^^^^^^^^^^^

The next level are instances of the
:class:`~psyplot.data.InteractiveBase` class. This abstract base
class provides an interface between the data and the visualization. Hence a
plotter (that's how we call instances of the :class:`Plotter` class) will deal
with the subclasses of the :class:`~psyplot.data.InteractiveBase`:

.. autosummary::

    ~psyplot.data.InteractiveArray
    ~psyplot.data.InteractiveList

Those classes (in particular the :class:`~psyplot.data.InteractiveArray`) keep
the reference to the base dataset to allow the update of the dataslice you are
plotting. The :class:`~psyplot.data.InteractiveList` class can be used in a
plotter for the visualization of multiple
:class:`~psyplot.data.InteractiveArray` instances (see for example the
:class:`psyplot.plotter.simple.LinePlotter` and
:class:`psyplot.plotter.maps.CombinedPlotter` classes).
Furthermore those data instances have a
:attr:`~psyplot.data.InteractiveBase.plotter` attribute that is usually
occupied by an instance of a :class:`Plotter` subclass.

Visualization objects
^^^^^^^^^^^^^^^^^^^^^
Each plotter class is the coordinator of several visualization options.
Thereby the :class:`~psyplot.plotter.Plotter` class itself contains only
the structural functionality for managing the formatoptions that do the
real work. The plotters for the real usage are defined in the modules of the
:mod:`~psyplot.plotter` package, in particular in the
:mod:`psyplot.plotter.maps` and :mod:`psyplot.plotter.lineplotter` modules.

Hence each :class:`~psyplot.data.InteractiveBase` instance is visualized by
exactly one :class:`Plotter` class. The initialization of such a setup is
straight forward. Just open a dataset, extract the right data
array and plot it

.. ipython::

    In [1]: from psyplot import InteractiveArray, open_dataset

    In [2]: from psyplot.plotter.maps import FieldPlotter

    In [3]: ds = open_dataset('demo.nc')

    In [4]: arr = InteractiveArray(ds.t2m[0, 0])

    @savefig docs_framework_plotter_demo.png width=4in
    In [5]: plotter = FieldPlotter(arr)

Now we created a plotter with all it's formatoptions:

.. ipython::

    In [6]: plotter

You can use the :meth:`~Plotter.show_keys`, :meth:`~Plotter.show_summaries` and
:meth:`~Plotter.show_docs` methods to have a look into the documentation into
the formatoptions or you simply use the bultin :func:`help` function for it::

    >>> help(plotter.clabel)

The update methods are the same as for the :class:`~psyplot.project.Project`
class. You can use the :meth:`InteractiveArray.update
<psyplot.data.InteractiveArray.update>` which updates the data and forwards
the formatoptions to the :meth:`Plotter.update` method.

.. note::

    Although you could now simply set a formatoption like you set an item for a
    dictionary via

    .. ipython::

        In [7]: plotter['clabel'] = 'my label'

    or equivalently

    .. ipython::

        In [7]: plotter.clabel = 'my label'

    However this would not change the plot! Instead you should use the
    :meth:`psyplot.plotter.Plotter.update` method, i.e.

    .. ipython::

        In [7]: plotter.update(clabel='my label')

Formatoptions
-------------
Formatoptions are the core of the visualization in the psyplot framework. They
conceptualy correspond to the basic :class:`matplotlib.artist.Artist` and
inherit from the abstract :class:`Formatoption` class. Each
plotter is set up through it's formatoptions where each formatoption has a
unique formatoption key inside the plotter. This formatoption key (e.g. 'title'
or 'clabel') is what is used for updating the plot etc. You can find more
information in :ref:`new_fmt`.