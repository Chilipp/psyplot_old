# -*- coding: utf-8 -*-
"""Module for fitting a linear model to the data

This module defines the :class:`LinRegPlotter` plotter class that can be used
to fit a linear model to the data and visualize it."""
from __future__ import division
import six
import inspect
from functools import partial
from itertools import islice, cycle, repeat
import numpy as np
from xarray import Coordinate, DataArray
import statsmodels.api as sm
from psyplot import rcParams
from psyplot.docstring import substitution_pattern, docstrings
from psyplot.compat.pycompat import range
from psyplot.data import InteractiveArray, InteractiveList, safe_list
from psyplot.plotter import Formatoption, START, Plotter, END
import psyplot.plotter.simple as psyps


class LinRegTranspose(psyps.Transpose):
    __doc__ = psyps.Transpose.__doc__

    priority = START


class XFitRange(psyps.Hist2DXRange):
    """
    Specify the range for the fit to use for the x-dimension

    This formatoption specifies the minimum and maximum of the fit
    in the x-dimension

    Possible types
    --------------
    %(LimitBase.possible_types.no_None)s

    Notes
    -----
    This formatoption always acts on the coordinate, no matter what the
    value of the :attr:`transpose` formatoption is

    See also
    --------
    yrange, line_xlim"""

    group = 'fit'

    @property
    def range(self):
        """The range for each of the curves"""
        return self._range

    @range.setter
    def range(self, value):
        if self.index_in_list is not None:
            self._range[self.index_in_list] = value
        else:
            self._range = value

    def update(self, value):
        if isinstance(self.raw_data, InteractiveList) and (
                self.index_in_list is None):
            self.range = [[] * len(self.raw_data)]
            for i in range(len(self.raw_data)):
                self.index_in_list = i
                super(XFitRange, self).update(value)
        elif not isinstance(self.raw_data, InteractiveList) and (
                self.index_in_list is not None):
            self._range = [[] * (self.index_in_list + 1)]
            super(XFitRange, self).update(value)
        else:
            super(XFitRange, self).update(value)

    def set_limit(self, *args):
        pass  # is set during update


class YFitRange(psyps.Hist2DYRange):
    """
    Specify the range for the fit to use for the y-dimension

    This formatoption specifies the minimum and maximum of the fit
    in the y-dimension

    Possible types
    --------------
    %(LimitBase.possible_types.no_None)s

    Notes
    -----
    This formatoption always acts on the coordinate, no matter what the
    value of the :attr:`transpose` formatoption is

    See also
    --------
    xrange"""

    group = 'fit'

    @property
    def range(self):
        """The range for each of the curves"""
        return self._range

    @range.setter
    def range(self, value):
        if self.index_in_list is not None:
            self._range[self.index_in_list] = value
        else:
            self._range = value

    def update(self, value):
        if isinstance(self.raw_data, InteractiveList) and (
                self.index_in_list is None):
            self.range = [[] * len(self.raw_data)]
            for i in range(len(self.raw_data)):
                self.index_in_list = i
                super(YFitRange, self).update(value)
        else:
            super(YFitRange, self).update(value)

    def set_limit(self, *args):
        pass  # is set during update


class XLineRange(XFitRange):
    """
    Specify how wide the range for the plot should be

    This formatoption specifies the range of the line to use

    Possible types
    --------------
    %(LimitBase.possible_types.no_None)s

    See Also
    --------
    xrange"""


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
    'poly<deg>'
        Make a polynomial fit of the order ``'<deg>'``
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

    dependencies = ['transpose', 'fix', 'xrange', 'yrange', 'coord',
                    'line_xlim']

    priority = START

    name = 'Change the fit method'

    data_dependent = True

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
        elif value.lower().startswith('poly'):
            self.model = partial(np.polyfit, deg=int(value[4:]), cov=True)
            self.method = 'poly'
        else:
            self.model = sm.RLM if value == 'robust' else sm.OLS
            self.method = 'statsmodels'
        transpose = self.transpose.value
        for i, da in enumerate(self.iter_raw_data):
            kwargs = self.get_kwargs(i)
            x, xname, y, yname = self.get_xy(i, da)
            x_line = self.get_xline(i)
            if self.coord.value is not None:
                da = self.coord.replace_coord(i)
            x_line, y_line, attrs, fit = self.make_fit(x, y, x_line=x_line,
                                                       **kwargs)
            if transpose:
                x_line, y_line = y_line, x_line
            attrs.update(da.attrs)
            coord_attrs = da.coords[da.dims[0]].attrs.copy()
            da_fit = InteractiveArray(
                data=y_line, dims=(xname, ), name=yname, attrs=attrs,
                coords={xname: Coordinate(xname, x_line, attrs=coord_attrs)},
                arr_name=da.arr_name).assign_coords(**self._get_other_coords(
                    da))
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

    def get_xline(self, i=0):
        """Get the x-data for the best fit line"""
        xrange = np.asarray(self.line_xlim.range)
        if xrange.ndim == 2:
            xmin, xmax = xrange[i]
        else:
            xmin, xmax = xrange
        return np.linspace(xmin, xmax, 100)

    def get_xy(self, i, da):
        if self.coord.value is not None:
            da = self.coord.replace_coord(i)
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
        xrange = np.array(self.xrange.range)
        yrange = np.array(self.yrange.range)
        if xrange.ndim == 1:
            xmin, xmax = xrange
            ymin, ymax = yrange
        else:
            xmin, xmax = xrange[i]
            ymin, ymax = yrange[i]
        mask = ~(np.isnan(x) | np.isnan(y)) & (x >= xmin) & (x <= xmax) & (
            y >= ymin) & (y <= ymax)
        return x[mask], xname, y[mask], yname

    def make_fit(self, x, y, x_line=None, **kwargs):
        if self.method == 'statsmodels':
            return self._statsmodel_fit(x, y, x_line, **kwargs)
        elif self.method == 'poly':
            return self._poly_fit(x, y, x_line, **kwargs)
        else:
            return self._scipy_curve_fit(x, y, x_line, **kwargs)

    def _scipy_curve_fit(self, x, y, x_line, **kwargs):
        from scipy.optimize import curve_fit
        params, pcov = curve_fit(self.model, x, y, **kwargs)
        if six.PY2:
            args = inspect.getargspec(self.model).args[1:]
        else:
            args = list(inspect.signature(self.model).parameters.keys())[1:]
        d = dict(zip(args, params))
        if pcov.size == 1:
            d[args[0] + '_err'] = np.sqrt(pcov)[0, 0]
        return x_line, self.model(x_line, *params), d, pcov

    def _poly_fit(self, x, y, x_line, **kwargs):
        params, pcov = self.model(x, y)
        d = dict(zip(('c%i' % i for i in range(len(params))),
                     params[::-1]))
        if pcov.size == 1:
            d['c0_err'] = np.sqrt(pcov)[0, 0]
        return x_line, np.poly1d(params)(x_line), d, pcov

    def _statsmodel_fit(self, x, y, x_line, fix=None):
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
            xmin, xmax = self.line_xlim.range
            x_line = np.linspace(xmin, xmax, 100)
        elif adjust:
            x_line = x_line - fix[0]
        d = dict(zip(['slope', 'intercept'], fit.params[::-1]))
        y_line = d.get('intercept', 0) + d['slope'] * x_line
        if adjust:
            x_line = x_line + fix[0]  # not += to make sure that Ci works fine
            y_line += fix[1]
            d['intercept'] = fix[1] - d['slope'] * fix[0]
        elif fix is not None:
            d['intercept'] = 0
        if hasattr(fit, 'rsquared'):
            d['rsquared'] = fit.rsquared
        return x_line, y_line, d, fit

    def _get_other_coords(self, raw_da):
        return {key: raw_da.coords[key]
                for key in set(raw_da.coords).difference(raw_da.dims)}


docstrings.delete_types('LineColors.possible_types', 'no_none', 'None')


class IdealLineColor(psyps.LineColors):
    """
    The colors of the ideal lines

    Possible types
    --------------
    None
        Let it be determined by the color cycle of the :attr:`color`
        formatoption
    %(LineColors.possible_types.no_none)s

    See Also
    --------
    ideal
    """

    parents = ['ideal']

    dependencies = ['color']

    priority = END

    def update(self, value):
        if self.ideal.value is not None:
            if value is None:
                value = self.color.color_cycle
            super(IdealLineColor, self).update(value)


class IdealLine(Formatoption):
    """
    Draw an ideal line of the fit

    Possible types
    --------------
    None
        Don't draw an ideal line
    list of floats
        The parameters for the line. If the :attr:`fit` formatoption is in
        ``'robust'`` or ``'fit'``, then the first value corresponds to the
        interception, the second to the slope. Otherwise the list corrensponds
        to the parameters as used in the fit function of the lines
    list of list of floats
        The same as above but with the specification for each array

    See Also
    --------
    id_color
    """

    dependencies = ['fit', 'id_color', 'plot']

    def initialize_plot(self, *args, **kwargs):
        self._plot = []
        super(IdealLine, self).initialize_plot(*args, **kwargs)

    def update(self, value):
        self.remove()
        if value is None:
            return
        # we update id_color here to make sure that the colors are only used
        # if necessary
        self.id_color.update(self.id_color.value)
        self._plot = []
        fit_type = self.fit.value
        if self.plot.value is None:
            linestyles = repeat('-')
        else:
            linestyles = cycle(safe_list(self.plot.value))
        for vals, da, c, ls in zip(cycle(value), self.iter_data,
                                   self.id_color.colors, linestyles):
            if da.ndim > 1:
                da = da[0]
            x = da.to_series().index.values
            if fit_type in ['robust', 'fit']:
                y = vals[0] + vals[1] * x
            else:
                y = fit_type(x, *vals)
            self._plot.extend(self.ax.plot(x, y, color=c, ls=ls))

    def remove(self):
        for artist in self._plot:
            artist.remove()
        self._plot = []


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

    name = 'Set the bootstrapping number to calculate the confidence interval'

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

    name = 'Draw a confidence interval'

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
            x, xname, y, yname = fit_fmt.get_xy(i, da)
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
                                   arr_name=da.arr_name).assign_coords(
                **self._get_other_coords(da_fit))
            self.set_data(new, i)
            new.attrs.update(da_fit.attrs)
            new.name = da.name

    def _get_other_coords(self, raw_da):
        return {key: raw_da.coords[key]
                for key in set(raw_da.coords).difference(raw_da.dims)}


class FitPointDensity(psyps.PointDensity):

    children = psyps.PointDensity.children + ['line_xlim']


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
    xrange = XFitRange('xrange')
    yrange = YFitRange('yrange')
    line_xlim = XLineRange('line_xlim')
    fit = LinearRegressionFit('fit')
    fix = FixPoint('fix')
    nboot = NBoot('nboot')
    ci = Ci('ci')
    id_color = IdealLineColor('id_color')
    ideal = IdealLine('ideal')


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
    line_xlim = XLineRange('line_xlim', index_in_list=0)
    precision = psyps.DataPrecision('precision', index_in_list=0)
    bins = psyps.HistBins('bins', index_in_list=0)
    normed = psyps.NormedHist2D('normed', index_in_list=0)
    density = FitPointDensity('density', index_in_list=0)

    # line plot formatoptions
    fit = LinearRegressionFit('fit', index_in_list=1)
    fix = FixPoint('fix', index_in_list=1)
    nboot = NBoot('nboot', index_in_list=1)
    ci = Ci('ci', index_in_list=1)
    lineplot = psyps.LinePlot('lineplot', index_in_list=1)
    error = psyps.ErrorPlot('error', index_in_list=1, plot='lineplot')
    erroralpha = psyps.ErrorAlpha('erroralpha', index_in_list=1)
    color = psyps.LineColors('color', index_in_list=1)
    legendlabels = psyps.LegendLabels('legendlabels', index_in_list=1)
    legend = psyps.Legend('legend', plot='lineplot', index_in_list=1)
    xlim = psyps.Xlim2D('xlim', index_in_list=0)
    ylim = psyps.Ylim2D('ylim', index_in_list=0)
    id_color = IdealLineColor('id_color', index_in_list=1)
    ideal = IdealLine('ideal', plot='lineplot', index_in_list=1)

for fmt in psyps.XYTickPlotter._get_formatoptions():
    fmto_cls = getattr(psyps.XYTickPlotter, fmt).__class__
    setattr(DensityRegPlotter, fmt, fmto_cls(fmt, index_in_list=1))

DensityRegPlotter.xlabel = psyps.Xlabel('xlabel', index_in_list=0)
DensityRegPlotter.ylabel = psyps.Ylabel('ylabel', index_in_list=0)
