.. _gallery_examples_plotter_simple_example_violin.ipynb:


Violin plot demo
================

This example shows you how to make a violin plot using the
``psyplot.project.ProjectPlotter.violinplot`` method.

.. code:: python

    import psyplot.project as psy

.. code:: python

    axes = iter(psy.multiple_subplots(2, 2, n=3))
    for var in ['t2m', 'u', 'v']:
        psy.plot.violinplot(
            'demo.nc',  # netCDF file storing the data
            name=var, # one plot for each variable
            t=range(5),  # one violin plot for each time step
            z=0, x=0,      # choose latitude and longitude as dimensions
            ylabel="{desc}",  # use the longname and units on the y-axis
            ax=next(axes),
            color='coolwarm', legend=False,
            xticklabels='%B %Y'  # choose xaxis labels to use month and year info,
        )
    violins = psy.gcp(True)
    violins.show()



.. image:: images/example_violin_0.png


.. code:: python

    violins.close(True, True)


.. only:: html

    .. container:: sphx-glr-download

        **Download python file:** :download:`example_violin.py`

        **Download IPython notebook:** :download:`example_violin.ipynb`


.. only:: html

    .. container:: sphx-glr-download

        **Download supplementary data:** :download:`demo.nc`
