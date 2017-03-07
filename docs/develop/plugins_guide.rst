.. _plugins:

How to implement your own plotters and plugins
==============================================

Creating plotters
-----------------

.. currentmodule:: psyplot.plotter

Implementing new plotters can be very easy or quite an effort depending on how
sophisticated you want to do it. In principle, you only have to implement the
:meth:`Formatoption.update` method and a default value. I.e., one simple
formatoption would be


.. ipython::

    In [1]: from psyplot.plotter import Formatoption, Plotter

    In [2]: class MyFormatoption(Formatoption):
       ...:     default = 'my text'
       ...:     def update(self, value):
       ...:         self.ax.text(0.5, 0.5, value, fontsize='xx-large')

together with a plotter

.. ipython::

    In [3]: class MyPlotter(Plotter):
       ...:     my_fmt = MyFormatoption('my_fmt')

and your done. Now you can make a simple plot

.. ipython::

    In [4]: from psyplot import open_dataset

    In [5]: ds = open_dataset('demo.nc')

    @savefig docs_demo_MyPlotter_simple.png width=4in
    In [6]: plotter = MyPlotter(ds.t2m)

However, if you're using the psyplot framework, you probably will be a bit more
advanced so let's talk about attributes and methods of the :class:`Formatoption`
class.

If you look into the documentation of the :class:`Formatoption` class, you find
quite a lot of attributes and methods which probably is a bit depressing and
confusing. But in principle, we can group them into 4 categories, the interface
to the data, to the plotter and to other formatoptions. Plus an additional
category for some Formatoption internals you definitely have to care about.

Interface for the plotter
^^^^^^^^^^^^^^^^^^^^^^^^^
The first interface is the one, that interfaces to the plotter. The most
important attributes in this group are the :attr:`~Formatoption.key`,
:attr:`~Formatoption.priority`, :attr:`~Formatoption.plot_fmt`,
:meth:`~Formatoption.initialize_plot` and most important the
:meth:`~Formatoption.update` method.

The :attr:`~Formatoption.key` is the unique key for the formatoption inside the
plotter. In our example above, we assign the ``'my_fmt'`` key to the
``MyFormatoption`` class in ``MyPlotter``. Hence, this key is defined when the
plotter class is defined and will be automatically assigned to the formatoption.

The next important attribute is the :attr:`priority` attribute. There are three
stages in the update of a plotter:

1. The stage with data manipulation. If formatoptions manipulate the data that
   shall be visualized (the :attr:`~Formatoption.data` attribute), those
   formatoptions are updated first. They have the :attr:`psyplot.plotter.START`
   priority
2. The stage of the plot. Formatoptions that influence how the data is
   visualized are updated here (e.g. the colormap or formatoptions that do the
   plotting). They have the :attr:`psyplot.plotter.BEFOREPLOTTING` priority.
3. The stage of the plot where additional informations are inserted. Here all
   the labels are updated, e.g. the title, xlabel, etc.. This is the default
   priority of the :class:`Formatoption.priority` attribute, the
   :attr:`psyplot.plotter.END` priority.

If there is any formatoption updated within the first two groups, the plot of
the plotter is updated. This brings us to the third important attribute, the
:attr:`~Formatoption.plot_fmt`. This boolean tells the plotter, whether the
corresponding formatoption is assumed to make a plot at the end of the second
stage (the :attr:`~psyplot.plotter.BEFOREPLOTTING` stage). If this attribute is
``True``, then the plotter will call the :meth:`Formatoption.make_plot` method
of the formatoption instance.

Finally, the :meth:`~Formatoption.initialize_plot` and
:meth:`~Formatoption.update` methods, this is were your contribution really is
required. The :meth:`~Formatoption.initialize_plot` method is called when the
plot is created for the first time, the :meth:`~Formatoption.update` method
when it is updated (the default implementation of the
:meth:`~Formatoption.initialize_plot` simply calls the
:meth:`~Formatoption.update` method). Implement theses methods in your
formatoption and make use of the interface to the
:ref:`data <fmt_data_interface>` and other
:ref:`formatoptions <fmt_fmt_interface>``.

.. _fmt_data_interface:

Interface to the data
^^^^^^^^^^^^^^^^^^^^^


.. _fmt_fmt_interface:

Interfacing to other formatoptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
