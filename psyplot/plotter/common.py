# -*- coding: utf-8 -*-
"""Axes Wrapper module of the nc2map Python module.

This module contains the necessary functions to wrap around the plt.subplots
function and store the subplot grid in the subplots."""
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy import ravel, array, append


def wrap_subplot(_instance, _num, _shape, *args, **kwargs):
    """Function that creates an AxesWrapper class around the given SubplotBase
    instance and saves the given number and shape to _AxesWrapper__num and
    _AxesWrapper__shape

    Parameters
    ----------
     _instance: matplotlib.axes.Axes
         Any instance (e.g. a subplot created with plt.subplots)
     num: int
         Number of the subplot in the plot (e.g. for
         ``plt.reset_geometry(x, y, num))``
    shape: tuple
        Shape of the subplots (e.g. from plt.subplots(x, y))

    Returns
    -------
    AxesWrapper
        an object being an instance of _instance.__class__
    """
    class AxesWrapper(type(_instance)):
        __metaclass__ = type
        __num = _num
        __shape = _shape
        __init_args = args
        __init_kwargs = kwargs
        __ax = _instance

        def __init__(self):
            pass

        def __getattribute__(self, attr):
            self_dict = object.__getattribute__(type(self), '__dict__')
            if attr in self_dict:
                return self_dict[attr]
            return getattr(_instance, attr)

        def __setattr__(self, attr, val):
            self_dict = object.__getattribute__(type(self), '__dict__')
            if attr in self_dict:
                object.__getattribute__(type(self), '__dict__')[attr] = val
            setattr(_instance, attr, val)

        def __eq__(self, value):
            return _instance == value

        def __ne__(self, value):
            return _instance != value

        def __repr__(self):
            return repr(_instance)

        def __str__(self):
            return str(_instance)

    return AxesWrapper()


def subplots(nrows=1, ncols=1, *args, **kwargs):
    """Same as maplotlib.pyplot.subplots function but returns a
    AxesWrapper instances instead of maplotlib.axes._subplots.AxesSubplot
    instances. AxesWrapper instances store the shape (nrows, ncols) in the
    _AxesWrapper__shape attribute and the number in the figure in the
    _AxesWrapper__num attribute
    """
    fig, subplots = plt.subplots(nrows, ncols, *args, **kwargs)
    kwargs.setdefault('num', fig.number)
    try:
        if subplots.ndim == 1:
            for i, ax in enumerate(subplots):
                subplots[i] = wrap_subplot(ax, i+1, (nrows, ncols), *args,
                                           **kwargs)
        else:
            counter = 1
            for i, axes in enumerate(subplots):
                for j, ax in enumerate(axes):
                    subplots[i, j] = wrap_subplot(
                        subplots[i, j], counter, (nrows, ncols), *args,
                        **kwargs)
                    counter += 1
    except AttributeError:
        subplots = wrap_subplot(subplots, 1, (nrows, ncols), *args, **kwargs)
    return fig, subplots


def multiple_subplots(ax=(1,1), n=1, *args, **kwargs):
    """Function to create subplots.

    Different from the :func:`~nc2map.subplots` function, this method creates
    multiple figures with the given shapes.

    Parameters
    ----------
    ax: tuple (x,y[,z]), subplot or list of subplots
        Default: (1, 1)

        - if matplotlib.axes.AxesSubplot instance or list of
            matplotlib.axes.AxesSubplot instances, those will be
            ravelled and returned
        - If ax is a tuple (x,y), figures will be created with x rows and
            y columns of subplots using the :func:`~nc2map.subplots`
            function. If ax is (x,y,z), only the first z subplots of each
            figure will be used.
    n: int
        number of subplots to create
    ``*args`` and ``**kwargs``
        anything that is passed to the :func:`~nc2map.subplots` function

    Returns
    -------
    list
        list of maplotlib.axes.SubplotBase instances"""
    logger = logging.getLogger(__name__)
    axes = array([])
    if ax is None:
        ax = (1, 1)
    if isinstance(ax, tuple):
        logger.debug("Creating new subplots with shape %s", ax)
        ax = list(ax)
        if len(ax) == 3:
            maxplots = ax[2]
            ax = ax[:2]
        else:
            maxplots = ax[0]*ax[1]
        kwargs.setdefault('figsize', [8.*ax[1]/max(ax), 6.5*ax[0]/max(ax)])
        for i in xrange(0, n, maxplots):
            fig, ax2 = subplots(ax[0], ax[1], *args, **kwargs)
            try:
                axes = append(axes, ax2.ravel()[:maxplots])
                for iax in xrange(maxplots, ax[0]*ax[1]):
                    fig.delaxes(ax2.ravel()[iax]._AxesWrapper__ax)
            except AttributeError:
                axes = append(axes, [ax2])
            if i + maxplots > n:
                for ax2 in axes[n:]:
                    fig.delaxes(ax2._AxesWrapper__ax)
                    axes = axes[:n]
    elif isinstance(ax, mpl.axes.SubplotBase):
        axes = append(axes, [ax])
    else:
        axes = append(axes, ravel(ax))
    logger.debug("Found %i subplots", len(axes))
    return axes.tolist()

def ax_property():
    """Returns a property that creates a new subplot if necessary"""

    def getx(self):
        """Axes instance of the plot"""
        if self._ax is None:
            fig, ax = subplots()
            try:
                fig.canvas.set_window_title(
                    'Figure %i: %s' % (fig.number, self.name))
            except AttributeError:
                pass
            self._ax = ax
            try:
                setattr(self.wind, 'ax', ax)
            except AttributeError:
                pass
        ax = self._ax
        try:
            return ax._AxesWrapper__ax
        except AttributeError:
            return ax

    def setx(self, value):
        self._ax = value

    def delx(self):
        self._ax = None

    doc = "Axes where the data of this instance in plotted on"

    return property(getx, setx, delx, doc)

subplots.__doc__ += plt.subplots.__doc__
