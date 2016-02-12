.. _gallery_examples_example_sharing.ipynb:


Sharing formatoptions
=====================

This example shows you the capabilities of sharing formatoptions and
what it is all about.

Within the psyplot framework you can easily manage multiple plots and
even have interaction between them. This is especially useful if want to
compare different variables.

This example requires the file ``'sharing_demo.nc'`` which contains one
variable for the temperature.

.. code:: python

    import psyplot.project as psy
    fname = 'sharing_demo.nc'

First we create 4 plots into one figure, one for each time step

.. code:: python

    maps = psy.plot.mapplot(fname, name='t2m', title='{dinfo}', ax=(2, 2), time=range(4))



.. image:: images/example_sharing_0.png


As you see, they have slightly different boundaries which can be very
annoying if we want to compare them. Therefore we can share the
boundaries of the colorbar. The corresponding formatoption is the
*bounds* formatoption

.. code:: python

    maps.share(keys='bounds')
    maps.show()



.. image:: images/example_sharing_1.png


Now the very first array (January 31st) shares the boundaries with all
the other. Furthermore it uses their data as well to calculate the
range.

The sharing of formatoptions works for every formatoption key and
formatoption groups.

.. code:: python

    maps[0].plotter.groups




.. parsed-literal::

    {'axes': 'Axes formatoptions',
     'colors': 'Color coding formatoptions',
     'labels': 'Label formatoptions',
     'masking': 'Masking formatoptions',
     'misc': 'Miscallaneous formatoptions',
     'plot': 'Plot formatoptions',
     'ticks': 'Axis tick formatoptions'}



Suppose for example, we want to work with only the last array but have
the color settings kept equal throughout each plot. For this we can
share the ``'colors'`` group of the formatoption. To do this, we should
first unshare the formatoptions currently the first one shares the
boundaries with the others.

.. code:: python

    maps.unshare(keys='bounds')
    # Now we share the color settings of the last one
    arr = maps[-1]
    maps[:-1].share(arr, keys='colors')

If we now update any of the color formatoptions of the last array, we
update them for all the others. However, the other formatoptions (in
this example the *projection*) keep untouched

.. code:: python

    arr.update(cmap='RdBu_r', projection='robin', time=4)
    maps.show()



.. image:: images/example_sharing_2.png


.. code:: python

    psy.gcp(True).close(True, True)


.. only:: html

    .. container:: sphx-glr-download

        **Download python file:** :download:`example_sharing.py`

        **Download IPython notebook:** :download:`example_sharing.ipynb`


.. only:: html

    .. container:: sphx-glr-download

        **Download supplementary data:** :download:`sharing_demo.nc`
