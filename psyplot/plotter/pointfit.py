# -*- coding: utf-8 -*-
import numpy as np
from xarray import Coordinate
from seaborn.algorithms import bootstrap
import statsmodels.formula.api as smf
from psyplot.data import InteractiveArray
from psyplot.plotter import Formatoption, START
from psyplot.plotter.simple import LinePlotter, Transpose


class LinFitTranspose(Transpose):
    __doc__ = Transpose.__doc__

    priority = START


class LinearFit(Formatoption):
    """
    Choose the linear fitting method

    This formatoption consists makes a linear fit of the data

    Possible types
    --------------
    'fit'
        make a linear fit
    'robust'
        make a robust linear fit
    None
        make no fit

    See Also
    --------
    force
    """

    dependencies = ['transpose', 'force']

    priority = START

    name = 'Disable or enable the fit'

    group = 'data'

    def update(self, value):
        if value is None:
            return
        elif value == 'robust':
            raise NotImplementedError("Not yet implemented!")
        force = self.force.value
        transpose = self.transpose.value
        for i, da in enumerate(self.data[:]):
            # make a dataframe with two columns out of the data array
            df = da.to_series().to_frame().reset_index()
            if transpose:
                x = da.name
                y = da.dims[0]
            else:
                x = da.da.dims[0]
                y = da.name
            da_fit, attrs = self.make_fit(df, x, y, force)
            if transpose:
                da_fit = da_fit.T
            da_fit.attrs.update(attrs)
            da_fit.attrs.update(da.attrs)
            da_fit.coords[da.dims[0]].attrs.update(
                da.coords[da.dims[0]].attrs)
            self.data[i] = da_fit

    def make_fit(self, df, x, y, force=None):
        """Make a linear fit of x to y"""
        adjust = force is not None and force[0] != 0 and force[1] != 0
        if adjust:
            df[x] = df[x] - force[0]
            df[y] = df[y] - force[1]
        formula = ('%s ~ %s' if force is not None else '%s ~ -1 + %s') % (
            y, x)
        fit = smf.ols(formula, df).fit()
        x_line = np.linspace(x.min(), x.max())
        d = fit.params.to_dict()
        y_line = d.get('Intercept', 0) + d[x] * x_line
        if adjust:
            x_line += force[0]
            y_line += force[1]
            d['Intercept'] = force[1] - fit.params[x] * force[0]
        return InteractiveArray(data=y_line, dims=(x, ), name=y,
                                coords={x: Coordinate(x, x_line)}), d


class Force(Formatoption):
    """
    Force the fit to go through a given point

    Possible types
    --------------
    None
        do not force the fit at all
    float f
        make a linear fit forced through ``(x, y) = (0, f)``
    tuple (x', y')
        make a linear fit forced through ``(x, y) = (x', y')``

    See Also
    --------
    fit"""

    priority = START

    name = 'Force the fit to go through a given point'

    group = 'data'

    def update(self, value):
        pass  # the work is done by the fit formatoption


class PointFitPlotter(LinePlotter):
    """A plotter to visualize the fit on the data"""

    transpose = LinFitTranspose('transpose')
    fit = LinearFit('fit')
    force = Force('force')
