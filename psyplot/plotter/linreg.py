# -*- coding: utf-8 -*-
"""Module for fitting a linear model to the data

This module defines the :class:`LinRegPlotter` plotter class that can be used
to fit a linear model to the data and visualize it."""
from __future__ import division
import six
import inspect
from itertools import islice, cycle
import numpy as np
from xarray import Coordinate, DataArray
import statsmodels.api as sm
from psyplot import rcParams
from psyplot.docstring import substitution_pattern
from psyplot.compat.pycompat import range
from psyplot.data import InteractiveArray, InteractiveList
from psyplot.plotter import Formatoption, START, Plotter
import psyplot.plotter.simple as psyps


class LinRegTranspose(psyps.Transpose):
    __doc__ = psyps.Transpose.__doc__

    priority = START


class LinearRegressionFit(Formatoption):
    """
    Choose the linear fitting method

    This formatoption consists makes a linear fit of the data

    Possible types
    --------------
    'fit'
        make a linear fit
    'robust'
        make a robust linear fit
    function
        A callable function that takes an x-array and a y-array as input and
        can be used for the :func:`scipy.optimize.curve_fit` function
    None
        make no fit

    Notes
    -----
    You can access the intercept, slope and rsquared by the correponding
    attribute. E.g.::

        >>> plotter.update(
        ...     legendlabels='%%(intercept)s + %%(slope)s * x, '
        ...     '$R^2$=%%(rsquared)s')

    See Also
    --------
    fix
    """

    dependencies = ['transpose', 'fix']

    priority = START

    name = 'Disable or enable the fit'

    group = 'fit'

    def __init__(self, *args, **kwargs):
        super(LinearRegressionFit, self).__init__(*args, **kwargs)
        self._kwargs = {}

    def update(self, value):
        self.fits = [None] * len(list(self.iter_data))
        if value is None:
            return
        elif callable(value):
            self.model = value
            self.method = 'curve_fit'
        else:
            self.model = sm.RLM if value == 'robust' else sm.OLS
            self.method = 'statsmodels'
        transpose = self.transpose.value
        for i, da in enumerate(self.iter_raw_data):
            kwargs = self.get_kwargs(i)
            x, xname, y, yname = self.get_xy(da)
            x_line, y_line, attrs, fit = self.make_fit(x, y, **kwargs)
            if transpose:
                x_line, y_line = y_line, x_line
            attrs.update(da.attrs)
            coord_attrs = da.coords[da.dims[0]].attrs.copy()
            da_fit = InteractiveArray(
                data=y_line, dims=(xname, ), name=yname, attrs=attrs,
                coords={xname: Coordinate(xname, x_line, attrs=coord_attrs)},
                arr_name=da.arr_name)
            self.fits[i] = fit
            da_fit.attrs.update(attrs)
            da_fit.attrs.update(da.attrs)
            da_fit.coords[da.dims[0]].attrs.update(
                da.coords[da.dims[0]].attrs)
            self.set_data(da_fit, i)

    def get_kwargs(self, i):
        '''Get the fitting kwargs for the line at index `i`'''
        ret = {}
        for key, val in self._kwargs.items():
            ret[key] = val[i]
        return ret

    def get_xy(self, da):
        if self.transpose.value:
            x = da.values
            xname = da.name
            y = da.coords[da.dims[0]].values
            yname = da.dims[0]
        else:
            x = da.coords[da.dims[0]].values
            xname = da.dims[0]
            y = da.values
            yname = da.name
        return x, xname, y, yname

    def make_fit(self, x, y, x_line=None, **kwargs):
        if self.method == 'statsmodels':
            return self._statsmodel_fit(x, y, x_line, **kwargs)
        else:
            return self._scipy_curve_fit(x, y, x_line, **kwargs)

    def _scipy_curve_fit(self, x, y, x_line=None, **kwargs):
        from scipy.optimize import curve_fit
        params, pcov = curve_fit(self.model, x, y, **kwargs)
        if x_line is None:
            x_line = np.linspace(x.min(), x.max(), 100)
        if six.PY2:
            d = dict(zip(inspect.getargspec(self.model).args[1:], params))
        else:
            args = list(inspect.signature(self.model).parameters.keys())[1:]
            d = dict(zip(args, params))
        return x_line, self.model(x_line, *params), d, pcov

    def _statsmodel_fit(self, x, y, x_line=None, fix=None):
        """Make a linear fit of x to y"""
        adjust = fix is not None and fix != [0, 0]
        if adjust:
            x = x - fix[0]
            y = y - fix[1]
        if fix is None:
            if x.ndim < 2:
                x = sm.add_constant(x)
            fit = self.model(y, x).fit()
        else:
            fit = self.model(y, x).fit()
        if x_line is None:
            x_line = np.linspace(x.min(), x.max(), 100)
        d = dict(zip(['slope', 'intercept'], fit.params[::-1]))
        y_line = d.get('intercept', 0) + d['slope'] * x_line
        if adjust:
            x_line += fix[0]
            y_line += fix[1]
            d['intercept'] = fix[1] - d['slope'] * fix[0]
        elif fix is not None:
            d['intercept'] = 0
        if hasattr(fit, 'rsquared'):
            d['rsquared'] = fit.rsquared
        return x_line, y_line, d, fit


class LinearRegressionFitCombined(LinearRegressionFit):
    __doc__ = substitution_pattern.sub('%\g<0>', LinearRegressionFit.__doc__)

    def set_data(self, data, i=None):
        '''Reimplemented to change the `arr_name` attribute of the given array
        '''
        data.arr_name += '_fit'
        return super(LinearRegressionFitCombined, self).set_data(data, i)


class FixPoint(Formatoption):
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

    group = 'fit'

    connections = ['fit']

    def update(self, value):
        if not callable(self.fit.value):
            n = len(list(self.iter_data))
            if len(value) != n:
                value = list(islice(cycle(value), 0, n))
            self.points = value
            self.fit._kwargs['fix'] = self.points
        else:
            self.fit._kwargs.pop('fix', None)


class NBoot(Formatoption):
    """
    Set the number of bootstrap resamples for the confidence interval

    Parameters
    ----------
    int
        Number of bootstrap resamples used to estimate the ``ci``. The default
        value attempts to balance time and stability; you may want to increase
        this value for "final" versions of plots.

    See Also
    --------
    ci
    """

    priority = START

    group = 'fit'

    def update(self, value):
        """Does nothing. The work is done by the :class:`Ci` formatoption"""
        pass


def bootstrap(x, y, func, n_boot, random_seed=None, **kwargs):
    """
    Simple bootstrap algorithm used to estimate the confidence interval

    This function is motivated by seaborns bootstrap algorithm
    :func:`seaborn.algorithms.bootstrap`
    """
    boot_dist = []
    n = len(x)
    rs = np.random.RandomState(
        random_seed if random_seed is not None else rcParams[
            'plotter.linreg.bootstrap.random_seed'])
    for i in range(int(n_boot)):
        resampler = rs.randint(0, n, n)
        x_ = x.take(resampler, axis=0)
        y_ = y.take(resampler, axis=0)
        boot_dist.append(func(x_, y_, **kwargs))
    return np.array(boot_dist)


def calc_ci(a, which=95, axis=None):
    """Return a quantile range from an array of values."""
    p = 50 - which / 2, 50 + which / 2
    return np.percentile(a, p, axis)


class Ci(Formatoption):
    """
    Draw a confidence interval

    Size of the confidence interval for the regression estimate. This will
    be drawn using translucent bands around the regression line. The
    confidence interval is estimated using a bootstrap; for large
    datasets, it may be advisable to avoid that computation by setting
    this parameter to None.

    Possible types
    --------------
    None
        Do not draw and calculate a confidence interval
    float
        A quantile between 0 and 100
    """

    dependencies = ['transpose', 'fit', 'nboot', 'fix']

    priority = START

    group = 'fit'

    def initialize_plot(self, *args, **kwargs):
        self.cis = []
        super(Ci, self).initialize_plot(*args, **kwargs)

    def update(self, value):
        def make_fit(x_, y_, **kwargs):
            return fit_fmt.make_fit(x_, y_, **kwargs)[1]
        self.remove()
        if value is None or self.fit.value is None:
            return
        fit_fmt = self.fit
        nboot = self.nboot.value
        for i, (da, da_fit) in enumerate(zip(self.iter_raw_data,
                                             self.iter_data)):
            x, xname, y, yname = fit_fmt.get_xy(da)
            coord = da_fit.coords[da_fit.dims[0]]
            x_line = coord.values
            kwargs = self.fit.get_kwargs(i)
            boot = bootstrap(x, y, func=make_fit, n_boot=nboot,
                             x_line=x_line, **kwargs)
            min_range, max_range = calc_ci(boot, value, axis=0).astype(
                da.dtype)
            ds = da_fit.to_dataset()
            ds['min_err'] = DataArray(
                min_range, coords={coord.name: coord}, dims=(coord.name, ),
                name='min_err')
            ds['max_err'] = DataArray(
                max_range, coords={coord.name: coord}, dims=(coord.name, ),
                name='max_err')
            new = InteractiveArray(ds.to_array(name=da.name), base=ds,
                                   arr_name=da.arr_name)
            self.set_data(new, i)
            new.attrs.update(da_fit.attrs)
            new.name = da.name


class LinRegPlotter(psyps.LinePlotter):
    """A plotter to visualize the fit on the data

    The most important formatoptions are the :attr:`fit` and :attr:`ci`
    formatoption. Otherwise this plotter behaves like the
    :class:`psyplot.plotter.simple.LinePlotter` plotter class"""

    _rcparams_string = ['plotter.linreg.']

    # only one variable is allowed because the error is determined through the
    # :attr:`ci` formatoption
    allowed_vars = 1

    transpose = LinRegTranspose('transpose')
    fit = LinearRegressionFit('fit')
    fix = FixPoint('fix')
    nboot = NBoot('nboot')
    ci = Ci('ci')


class DensityRegPlotter(psyps.ScalarCombinedBase, psyps.DensityPlotter,
                        LinRegPlotter):
    """A plotter that visualizes the density of points together with a linear
    regression"""

    _rcparams_string = ['plotter.densityreg.']

    def _set_data(self, *args, **kwargs):
        Plotter._set_data(self, *args, **kwargs)
        self._plot_data = InteractiveList(
            [InteractiveArray([]), InteractiveArray([])])

    # scalar (density) plot formatoptions
    cbar = psyps.Cbar('cbar')
    plot = psyps.Plot2D('plot', index_in_list=0)
    xrange = psyps.Hist2DXRange('xrange', index_in_list=0)
    yrange = psyps.Hist2DYRange('yrange', index_in_list=0)
    precision = psyps.DataPrecision('precision', index_in_list=0)
    bins = psyps.HistBins('bins', index_in_list=0)
    normed = psyps.NormedHist2D('normed', index_in_list=0)
    density = psyps.PointDensity('density', index_in_list=0)

    # line plot formatoptions
    fit = LinearRegressionFit('fit', index_in_list=1)
    fix = FixPoint('fix', index_in_list=1)
    nboot = NBoot('nboot', index_in_list=1)
    ci = Ci('ci', index_in_list=1)
    lineplot = psyps.LinePlot('lineplot', index_in_list=1)
    error = psyps.ErrorPlot('error', index_in_list=1)
    erroralpha = psyps.ErrorAlpha('erroralpha', index_in_list=1)
    color = psyps.LineColors('color', index_in_list=1)
    legendlabels = psyps.LegendLabels('legendlabels', index_in_list=1)
    legend = psyps.Legend('legend', plot='lineplot', index_in_list=1)
    xlim = psyps.Xlim2D('xlim', index_in_list=0)
    ylim = psyps.Ylim2D('ylim', index_in_list=0)

for fmt in psyps.XYTickPlotter._get_formatoptions():
    fmto_cls = getattr(psyps.XYTickPlotter, fmt).__class__
    setattr(DensityRegPlotter, fmt, fmto_cls(fmt, index_in_list=1))

DensityRegPlotter.xlabel = psyps.Xlabel('xlabel', index_in_list=0)
DensityRegPlotter.ylabel = psyps.Ylabel('ylabel', index_in_list=0)
