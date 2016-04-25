import six
from abc import abstractproperty, abstractmethod
from itertools import chain, starmap, cycle
from pandas import date_range, datetools, to_datetime
import matplotlib as mpl
import matplotlib.axes
from matplotlib.ticker import FormatStrFormatter, FixedLocator, FixedFormatter
from matplotlib.dates import DateFormatter, AutoDateFormatter
import matplotlib.colors as mcol
import numpy as np
from psyplot.docstring import docstrings, dedent
from psyplot.warning import warn, PsyPlotRuntimeWarning
from psyplot.plotter import (
    Plotter, Formatoption, BEFOREPLOTTING, DictFormatoption, END, rcParams)
from psyplot.plotter.baseplotter import (
    BasePlotter, TextBase, label_size, label_weight, label_props, MaskLess,
    MaskGreater, MaskBetween, MaskLeq, MaskGeq)
from psyplot.plotter.colors import get_cmap
from psyplot.data import InteractiveList, isstring
from psyplot.compat.pycompat import map, zip, range
from psyplot.config.rcsetup import validate_color, validate_float, safe_list


def round_to_05(n, exp=None, mode='s'):
    """
    Round to the next 0.5-value.

    This function applies the round function `func` to round `n` to the
    next 0.5-value with respect to its exponent with base 10 (i.e.
    1.3e-4 will be rounded to 1.5e-4) if `exp` is None or with respect
    to the given exponent in `exp`.

    Parameters
    ----------
    n: numpy.ndarray
        number to round
    exp: int or numpy.ndarray
        Exponent for rounding. If None, it will be computed from `n` to be the
        exponents for base 10.
    mode: {'s', 'l'}
        rounding mode. If 's', it will be rounded to value whose absolute
        value is below `n`, if 'l' it will rounded to the value whose absolute
        value is above `n`.

    Returns
    -------
    numpy.ndarray
        rounded `n`

    Examples
    --------
    The effects of the different parameters are show in the example below::

        >>> from psyplot.plotter.simple import round_to_05
        >>> a = [-100.3, 40.6, 8.7, -0.00023]
        >>>round_to_05(a, mode='s')
        array([ -1.00000000e+02,   4.00000000e+01,   8.50000000e+00,
                -2.00000000e-04])

        >>> round_to_05(a, mode='l')
        array([ -1.50000000e+02,   4.50000000e+01,   9.00000000e+00,
                -2.50000000e-04])"""
    n = np.asarray(n)
    if exp is None:
        exp = np.floor(np.log10(np.abs(n)))  # exponent for base 10
    ntmp = np.abs(n)/10.**exp  # mantissa for base 10
    if mode == 's':
        n1 = ntmp
        s = 1.
        n2 = nret = np.floor(ntmp)
    else:
        n1 = nret = np.ceil(ntmp)
        s = -1.
        n2 = ntmp
    return np.where(n1 - n2 > 0.5, np.sign(n)*(nret + s*0.5)*10.**exp,
                    np.sign(n)*nret*10.**exp)


class Grid(Formatoption):
    """
    Display the grid

    Show the grid on the plot with the specified color.


    Possible types
    --------------
    None
        If the grid is currently shown, it will not be displayed any longer. If
        the grid is not shown, it will be drawn
    bool
        If True, the grid is displayed with the automatic settings (usually
        black)
    string, tuple.
        Defines the color of the grid.

    Notes
    -----
    %(colors)s"""

    group = 'axes'

    name = 'Grid lines'

    def update(self, value):
        try:
            value = validate_color(value)
            self.ax.grid(color=value)
        except (ValueError, TypeError):
            self.ax.grid(value)


class AxisColor(DictFormatoption):
    """
    Color the x- and y-axes

    This formatoption colors the left, right, bottom and top axis bar.

    Possible types
    --------------
    dict
        Keys may be one of {'right', 'left', 'bottom', 'top'},  the values can
        be any valid color or None.

    Notes
    -----
    %(colors)s"""

    group = 'axes'

    name = 'Color of x- and y-axes'

    def initialize_plot(self, value):
        positions = ['right', 'left', 'bottom', 'top']
        #: :class:`dict` storing the default linewidths
        self.default_lw = dict(zip(positions, map(
            lambda pos: self.ax.spines[pos].get_linewidth(), positions)))
        self.update(value)

    def update(self, value):
        for pos, color in six.iteritems(value):
            spine = self.ax.spines[pos]
            spine.set_color(color)
            if color is not None and spine.get_linewidth() == 0.0:
                spine.set_linewidth(1.0)
            elif color is None:
                spine.set_color(mpl.rcParams['axes.edgecolor'])
                spine.set_linewidth(self.default_lw[pos])


class TicksManagerBase(Formatoption):
    """
    Abstract base class for formatoptions handling ticks"""

    @abstractmethod
    def update_axis(self, val):
        pass


@docstrings.get_sectionsf('TicksManager')
class TicksManager(TicksManagerBase, DictFormatoption):
    """
    Abstract base class for ticks formatoptions controlling major and minor
    ticks

    This formatoption simply serves as a base that allows the simultaneous
    managment of major and minor ticks

    Possible types
    --------------
    dict
        A dictionary with the keys ``'minor'`` and (or) ``'major'`` to specify
        which ticks are managed. If the given value is not a dictionary with
        those keys, it is put into a dictionary with the key determined by the
        rcParams ``'ticks.which'`` key (usually ``'major'``).
        The values in the dictionary can be one types below."""

    group = 'ticks'

    def update(self, value):
        for which, val in six.iteritems(value):
            self.which = which
            self.update_axis(val)


@docstrings.get_sectionsf('DataTicksCalculator')
class DataTicksCalculator(Formatoption):
    """
    Abstract base formatoption to calculate ticks and bounds from the data

    Possible types
    --------------
    numeric array
        specifies the ticks manually
    str or list [str, ...]
        Automatically determine the ticks corresponding to the data. The given
        string determines how the ticks are calculated. If not a single string
        but a list, the second value determines the number of ticks (see
        below). A string can be one of the following:

        data
            plot the ticks exactly where the data is.
        mid
            plot the ticks in the middle of the data.
        rounded
            Sets the minimum and maximum of the ticks to the rounded data
            minimum or maximum. Ticks are rounded to the next 0.5 value with
            to the difference between data max- and minimum. The minimal tick
            will always be lower or equal than the data minimum, the maximal
            tick will always be higher or equal than the data maximum.
        roundedsym
            Same as `rounded` above but the ticks are chose such that they are
            symmetric around zero
        minmax
            Uses the minimum as minimal tick and maximum as maximal tick
        sym
            Same as minmax but symmetric around zero"""

    data_dependent = True

    @property
    def array(self):
        """The numpy array of the data"""
        if self.shared:
            return np.concatenate(
                [self.data.values[~np.isnan(self.data.values)]] + [
                    fmto.array for fmto in self.shared])
        return self.data.values[~np.isnan(self.data.values)]

    def _data_ticks(self, step=None):
        step = step or 1
        """Array of ticks that match exactly the data"""
        return np.unique(self.array)[::step]

    def _mid_data_ticks(self, step=None):
        step = step or 1
        """Array of ticks in the middle between the data points"""
        arr = np.unique(self.array)
        return ((arr[:-1] + arr[1:])/2.)[::step]

    def _calc_vmin_vmax(self, percmin=None, percmax=None):
        def minmax(arr):
            return [arr.min(), arr.max()]
        percentiles = []
        if not self.shared:
            arr = self.array
        else:
            # np.concatenate all arrays if any of the percentiles are required
            if percmin is not None or percmax is not None:
                arr = np.concatenate(tuple(chain(
                    [self.array], (fmto.array for fmto in self.shared))))
            # np.concatenate only min and max-values instead of the full arrays
            else:
                arr = np.concatenate(tuple(map(minmax, chain(
                    [self.array], (fmto.array for fmto in self.shared)))))
        if not percmin:
            vmin = arr.min()
        else:
            percentiles.append(percmin)
        if percmax is None or percmax == 100:
            vmax = arr.max()
        else:
            percentiles.append(percmax)
        if percentiles:
            percentiles = iter(np.percentile(arr, percentiles))
            if percmin:
                vmin = next(percentiles)
            if percmax and percmax < 100:
                vmax = next(percentiles)
        return vmin, vmax

    @staticmethod
    def _round_min_max(vmin, vmax):
        exp = np.floor(np.log10(abs(vmax - vmin)))
        larger = round_to_05([vmin, vmax], exp, mode='l')
        smaller = round_to_05([vmin, vmax], exp, mode='s')
        return min([larger[0], smaller[0]]), max([larger[1], smaller[1]])

    def _rounded_ticks(self, N=None, *args, **kwargs):
        N = N or 11
        vmin, vmax = self._round_min_max(
            *self._calc_vmin_vmax(*args, **kwargs))
        return np.linspace(vmin, vmax, N, endpoint=True)

    def _roundedsym_ticks(self, N=None, *args, **kwargs):
        N = N or 10
        vmax = max(map(abs, self._round_min_max(
            *self._calc_vmin_vmax(*args, **kwargs))))
        vmin = -vmax
        return np.linspace(vmin, vmax, N, endpoint=True)

    def _data_minmax_ticks(self, N=None, *args, **kwargs):
        N = N or 11
        vmin, vmax = self._calc_vmin_vmax(*args, **kwargs)
        return np.linspace(vmin, vmax, N, endpoint=True)

    def _data_symminmax_ticks(self, N=None, *args, **kwargs):
        N = N or 10
        vmax = max(map(abs, self._calc_vmin_vmax(*args, **kwargs)))
        vmin = -vmax
        return np.linspace(vmin, vmax, N, endpoint=True)

    def __init__(self, *args, **kwargs):
        super(DataTicksCalculator, self).__init__(*args, **kwargs)
        self.calc_funcs = {
            'data': self._data_ticks,
            'mid': self._mid_data_ticks,
            'rounded': self._rounded_ticks,
            'roundedsym': self._roundedsym_ticks,
            'minmax': self._data_minmax_ticks,
            'sym': self._data_symminmax_ticks,
            }


@docstrings.get_sectionsf('TicksBase')
class TicksBase(TicksManagerBase, DataTicksCalculator):
    """
    Abstract base class for calculating ticks

    Possible types
    --------------
    None
        use the default ticks
    int
        for an integer *i*, only every *i-th* tick of the default ticks are
        used"""

    dependencies = ['transpose', 'plot']

    group = 'ticks'

    @abstractproperty
    def axis(self):
        pass

    def __init__(self, *args, **kwargs):
        super(TicksBase, self).__init__(*args, **kwargs)
        self.default_locators = {}

    def initialize_plot(self, value):
        self.set_default_locators()
        self.update(value)

    def update_axis(self, value):
        which = self.which
        if value is None:
            self.set_locator(self.default_locators[which])
        elif isinstance(value, int):
            return self._reduce_ticks(value)
        elif isinstance(value[0], six.string_types):
            return self.set_ticks(self.calc_funcs[value[0]](*value[1:]))
        elif isinstance(value, tuple):
            steps = 11 if len(value) == 2 else value[3]
            self.set_ticks(np.linspace(value[0], value[1], steps,
                                       endpoint=True))
        else:
            self.set_ticks(value)

    def set_ticks(self, value):
        self.axis.set_ticks(value, minor=self.which == 'minor')

    def get_locator(self):
        return getattr(self.axis, 'get_%s_locator' % self.which)()

    def set_locator(self, locator):
        """Sets the locator corresponding of the axis

        Parameters
        ----------
        locator: matplotlib.ticker.Locator
            The locator to set
        which: {None, 'minor', 'major'}
            Specify which locator shall be set. If None, it will be taken from
            the :attr:`which` attribute"""
        getattr(self.axis, "set_%s_locator" % self.which)(locator)

    def set_default_locators(self, which=None):
        """Sets the default locator that is used for updating to None or int

        Parameters
        ----------
        which: {None, 'minor', 'major'}
            Specify which locator shall be set"""
        if which is None or which == 'minor':
            self.default_locators['minor'] = self.axis.get_minor_locator()
        if which is None or which == 'major':
            self.default_locators['major'] = self.axis.get_major_locator()

    def _reduce_ticks(self, i):
        loc = self.default_locators[self.which]
        self.set_locator(FixedLocator(loc()[::i]))


@docstrings.get_sectionsf('DtTicksBase')
class DtTicksBase(TicksBase, TicksManager):
    """
    Abstract base class for x- and y-tick formatoptions

    Possible types
    --------------
    %(TicksManager.possible_types)s
    %(TicksBase.possible_types)s
    %(DataTicksCalculator.possible_types)s
        hour
            draw ticks every hour
        day
            draw ticks every day
        week
            draw ticks every week
        month, monthend, monthbegin
            draw ticks in the middle, at the end or at the beginning of each
            month
        year, yearend, yearbegin
            draw ticks in the middle, at the end or at the beginning of each
            year

        For data, mid, hour, day, week, month, etc., the optional second value
        can be an integer i determining that every i-th data point shall be
        used (by default, it is set to 1). For rounded, roundedsym, minmax and
        sym, the second value determines the total number of ticks (defaults to
        11)."""

    def __init__(self, *args, **kwargs):
        super(DtTicksBase, self).__init__(*args, **kwargs)
        self.calc_funcs.update({
            'hour': self._frequent_ticks('H'),
            'day': self._frequent_ticks('D'),
            'week': self._frequent_ticks(datetools.Week()),
            'month': self._mid_dt_ticks('M'),
            'monthend': self._frequent_ticks(
                datetools.MonthEnd(), onset=datetools.MonthBegin()),
            'monthbegin': self._frequent_ticks(
                datetools.MonthBegin(), onset=datetools.MonthBegin(),
                offset=datetools.MonthBegin()),
            'year': self._mid_dt_ticks(datetools.YearBegin()),
            'yearend': self._frequent_ticks(
                datetools.YearEnd(), onset=datetools.YearBegin()),
            'yearbegin': self._frequent_ticks(
                datetools.YearBegin(), onset=datetools.YearBegin(),
                offset=datetools.YearBegin())})

    def update(self, value):
        value = value or {'minor': None, 'major': None}
        super(DtTicksBase, self).update(value)

    @property
    def dtdata(self):
        """The np.unique :attr:`data` as datetime objects"""
        data = self.data
        # do nothing if the data is a pandas.Index without time informations
        # or not a pandas.Index
        if not getattr(data, 'is_all_dates', None):
            warn("Could not convert time informations for %s ticks with "
                 "object %r." % (self.key, type(data)), logger=self.logger)
            return None
        else:
            return data

    def _frequent_ticks(self, freq, onset=None, offset=None):
        def func(step=None):
            step = step or 1
            data = self.dtdata
            if data is None:
                return
            mindata = data.min() if onset is None else data.min() - onset
            maxdata = data.max() if offset is None else data.max() + offset
            return date_range(
                mindata, maxdata, freq=freq)[::step].to_pydatetime()
        return func

    def _mid_dt_ticks(self, freq):
        def func(step=None):
            step = step or 1
            data = self.dtdata
            if data is None:
                return
            data = date_range(
                data.min(), data.max(), freq=freq).to_pydatetime()
            data[:-1] += (data[1:] - data[:-1])/2
            return data[:-1:step]
        return func


class XTicks(DtTicksBase):
    """
    Modify the x-axis ticks

    Possible types
    --------------
    %(DtTicksBase.possible_types)s

    Examples
    --------
    Plot 11 ticks over the whole data range::

        >>> plotter.update(xticks='rounded')

    Plot 7 ticks over the whole data range where the maximal and minimal
    tick matches the data maximum and minimum::

        >>> plotter.update(xticks=['minmax', 7])

    Plot ticks every year and minor ticks every month::

        >>> plotter.update(xticks={'major': 'year', 'minor': 'month'})

    See Also
    --------
    xticklabels, ticksize, tickweight, xtickprops, yticks"""

    children = TicksBase.children + ['yticks']

    name = 'Location of the x-Axis ticks'

    @property
    def axis(self):
        return self.ax.xaxis

    @property
    def data(self):
        df = super(XTicks, self).data.to_dataframe()
        if self.transpose.value:
            return df
        else:
            return df.index

    def initialize_plot(self, *args, **kwargs):
        super(XTicks, self).initialize_plot(*args, **kwargs)
        self.transpose.swap_funcs['ticks'] = self._swap_ticks

    def _swap_ticks(self):
        xticks = self
        yticks = self.yticks
        old_xlocators = xticks.default_locators
        xticks.default_locators = yticks.default_locators
        yticks.default_locators = old_xlocators
        old_xval = self.value
        with self.plotter.no_validation:
            self.plotter[self.key] = self.yticks.value
            self.plotter[self.yticks.key] = old_xval


class YTicks(DtTicksBase):
    """
    Modify the y-axis ticks

    Possible types
    --------------
    %(DtTicksBase.possible_types)s

    See Also
    --------
    yticklabels, ticksize, tickweight, ytickprops
    xticks: for possible examples"""

    name = 'Location of the y-Axis ticks'

    @property
    def axis(self):
        return self.ax.yaxis

    @property
    def data(self):
        df = super(YTicks, self).data.to_dataframe()
        if self.transpose.value:
            return df.index
        else:
            return df


@docstrings.get_sectionsf('TickLabelsBase')
class TickLabelsBase(TicksManagerBase):
    """
    Abstract base class for ticklabels

    Possible types
    --------------
    str
        A formatstring like ``'%%Y'`` for plotting the year (in the case that
        time is shown on the axis) or '%%i' for integers
    array
        An array of strings to use for the ticklabels"""

    dependencies = ['transpose']

    group = 'ticks'

    @abstractproperty
    def axis(self):
        """The axis on the axes to modify the ticks of"""
        pass

    def __init__(self, *args, **kwargs):
        super(TickLabelsBase, self).__init__(*args, **kwargs)
        self.default_formatters = {}

    def initialize_plot(self, value):
        self.set_default_formatters()
        self.update(value)

    def update_axis(self, value):
        if value is None:
            self.set_formatter(self.default_formatters['major'])
        elif isinstance(value, six.string_types):
            self.set_stringformatter(value)
        else:
            ticks = self.axis.get_ticklocs(minor=self.which == 'minor')
            if len(ticks) != len(value):
                warn("[%s] - Length of ticks (%i) and ticklabels (%i)"
                     "do not match!" % (self.key, len(ticks), len(value)),
                     logger=self.logger)
            self.set_ticklabels(value)

    def set_stringformatter(self, s):
        default_formatter = self.default_formatters['major']
        if isinstance(default_formatter, AutoDateFormatter):
            self.set_formatter(DateFormatter(s))
        else:
            self.set_formatter(FormatStrFormatter(s))

    def set_ticklabels(self, labels):
        """Sets the given tick labels"""
        self.set_formatter(FixedFormatter(labels))

    @abstractmethod
    def set_formatter(self, formatter):
        """Sets a given formatter"""
        pass

    @abstractmethod
    def set_default_formatters(self):
        """Sets the default formatters that is used for updating to None"""
        pass


class TickLabels(TickLabelsBase, TicksManager):

    def update(self, value):
        if (getattr(self, self.key.replace('label', '')).value.get(
                 'minor') is not None and
                'minor' not in self.value):
            items = chain(six.iteritems(value), [('minor', None)])
        else:
            items = six.iteritems(value)
        super(TickLabels, self).update(dict(items))

    def set_default_formatters(self, which=None):
        """Sets the default formatters that is used for updating to None

        Parameters
        ----------
        which: {None, 'minor', 'major'}
            Specify which locator shall be set"""
        if which is None or which == 'minor':
            self.default_formatters['minor'] = self.axis.get_minor_formatter()
        if which is None or which == 'major':
            self.default_formatters['major'] = self.axis.get_major_formatter()

    def set_formatter(self, formatter, which=None):
        which = which or self.which
        getattr(self.axis, 'set_%s_formatter' % which)(formatter)


class XTickLabels(TickLabels):
    """
    Modify the x-axis ticklabels

    Possible types
    --------------
    %(TicksManager.possible_types)s
    %(TickLabelsBase.possible_types)s

    See Also
    --------
    xticks, ticksize, tickweight, xtickprops, yticklabels"""

    dependencies = TickLabelsBase.dependencies + ['xticks', 'yticklabels']

    name = 'x-xxis Ticklabels'

    @property
    def axis(self):
        return self.ax.xaxis

    def initialize_plot(self, *args, **kwargs):
        super(XTickLabels, self).initialize_plot(*args, **kwargs)
        self.transpose.swap_funcs['ticklabels'] = self._swap_ticklabels

    def _swap_ticklabels(self):
        xticklabels = self
        yticklabels = self.yticklabels
        old_xformatters = xticklabels.default_formatters
        xticklabels.default_formatters = yticklabels.default_formatters
        yticklabels.default_formatters = old_xformatters
        old_xval = self.value
        with self.plotter.no_validation:
            self.plotter[self.key] = self.yticklabels.value
            self.plotter[self.yticklabels.key] = old_xval


class YTickLabels(TickLabels):
    """
    Modify the y-axis ticklabels

    Possible types
    --------------
    %(TicksManager.possible_types)s
    %(TickLabelsBase.possible_types)s

    See Also
    --------
    yticks, ticksize, tickweight, ytickprops, xticklabels"""

    dependencies = TickLabelsBase.dependencies + ['yticks']

    name = 'y-xxis ticklabels'

    @property
    def axis(self):
        return self.ax.yaxis


class BarXTickLabels(XTickLabels):

    __doc__ = XTickLabels.__doc__

    dependencies = XTickLabels.dependencies + ['plot']

    def set_stringformatter(self, s):
        if not self.transpose.value and self.plot.value is not None:
            index = self.data.to_dataframe().index
            if index.is_all_dates:
                xticks = self.ax.get_xticks(self.which == 'minor')
                arr = list(map(lambda t: t.toordinal(),
                               to_datetime(index[xticks.astype(int)])))
                self.ax.set_xticklabels(list(map(DateFormatter(s), arr)))
                return
        super(BarXTickLabels, self).set_stringformatter(s)


class BarYTickLabels(YTickLabels):

    __doc__ = YTickLabels.__doc__

    dependencies = YTickLabels.dependencies + ['plot']

    def set_stringformatter(self, s):
        if not self.transpose.value and self.plot.value is not None:
            index = self.data.to_dataframe().index
            if index.is_all_dates:
                yticks = self.ax.get_yticks(self.which == 'minor')
                arr = list(map(lambda t: t.toordinal(),
                               to_datetime(index[yticks.astype(int)])))
                self.ax.set_yticklabels(list(map(DateFormatter(s), arr)))
                return
        super(BarYTickLabels, self).set_stringformatter(s)


class TicksOptions(TicksManagerBase):
    """Base class for ticklabels options that apply for x- and y-axis"""

    def update(self, value):
        for which, val in six.iteritems(value):
            for axis in [self.ax.xaxis, self.ax.yaxis]:
                self.which = which
                self.axis = axis
                self.update_axis(val)


class TickSizeBase(TicksOptions):
    """Abstract base class for modifying tick sizes"""

    def update_axis(self, value):
        self.axis.set_tick_params(which=self.which, labelsize=value)


class TickSize(TickSizeBase, TicksOptions, DictFormatoption):
    """
    Change the ticksize of the ticklabels

    Possible types
    --------------
    %(TicksManager.possible_types)s
    %(fontsizes)s

    See Also
    --------
    tickweight, xtickprops, ytickprops"""

    name = 'Font size of the ticklabels'


class TickWeightBase(TicksOptions):
    """Abstract base class for modifying font weight of ticks"""

    def update_axis(self, value):
        for t in self.axis.get_ticklabels(which=self.which):
            t.set_weight(value)


class TickWeight(TickWeightBase, TicksOptions, DictFormatoption):
    """
    Change the fontweight of the ticks

    Possible types
    --------------
    %(TicksManager.possible_types)s
    %(fontweights)s

    See Also
    --------
    ticksize, xtickprops, ytickprops"""

    name = 'Font weight of the ticklabels'


@docstrings.get_sectionsf('TickPropsBase')
class TickPropsBase(TicksManagerBase):
    """
    Abstract base class for tick parameters

    Possible types
    --------------
    dict
        Items may be anything of the :func:`matplotlib.pyplot.tick_params`
        function"""

    children = ['ticksize']

    @abstractproperty
    def axisname(self):
        """The name of the axis (either 'x' or 'y')"""
        pass

    def update_axis(self, value):
        value = value.copy()
        default = self.default
        if 'major' in default or 'minor' in default:
            default = default.get(self.which, {})
        for key, val in chain(
                default.items(), mpl.rcParams.find_all(
                    self.axisname + 'tick\.%s\.\w' % self.which).items()):
            value.setdefault(key.split('.')[-1], val)

        if float('.'.join(mpl.__version__.split('.')[:2])) >= 1.5:
            value.pop('visible', None)

        if 'labelsize' not in value:
            if isinstance(self.ticksize.value, dict):
                labelsize = self.ticksize.value.get(
                    self.which, mpl.rcParams[self.axisname + 'tick.labelsize'])
            else:
                labelsize = self.ticksize.value
            self.axis.set_tick_params(
                which=self.which, labelsize=labelsize, **value)
            self.axis.set_tick_params(
                which=self.which, labelsize=labelsize, **value)
        else:
            self.axis.set_tick_params(which=self.which, **value)
            self.axis.set_tick_params(which=self.which, **value)


@docstrings.get_sectionsf('XTickProps')
class XTickProps(TickPropsBase, TicksManager, DictFormatoption):
    """
    Specify the x-axis tick parameters

    This formatoption can be used to make a detailed change of the ticks
    parameters on the x-axis.

    Possible types
    --------------
    %(TicksManager.possible_types)s
    %(TickPropsBase.possible_types)s

    See Also
    --------
    xticks, yticks, ticksize, tickweight, ytickprops"""

    axisname = 'x'

    name = 'Font properties of the x-ticklabels'

    @property
    def axis(self):
        return self.ax.xaxis



class YTickProps(XTickProps):
    """
    Specify the y-axis tick parameters

    This formatoption can be used to make a detailed change of the ticks
    parameters of the y-axis.

    Possible types
    --------------
    %(XTickProps.possible_types)s

    See Also
    --------
    xticks, yticks, ticksize, tickweight, xtickprops"""

    axisname = 'y'

    name = 'Font properties of the y-ticklabels'

    @property
    def axis(self):
        return self.ax.xaxis



class Xlabel(TextBase, Formatoption):
    """
    Set the x-axis label

    Set the label for the x-axis.
    %(replace_note)s

    Possible types
    --------------
    str
        The text for the :func:`~matplotlib.pyplot.xlabel` function.

    See Also
    --------
    xlabelsize, xlabelweight, xlabelprops"""

    children = ['transpose', 'ylabel']

    name = 'x-axis label'

    def initialize_plot(self, value):
        self.transpose.swap_funcs['labels'] = self._swap_labels
        arr = self.transpose.get_x(self.data)
        attrs = self.get_enhanced_attrs(arr)
        self._texts = [self.ax.set_xlabel(self.replace(
            value, self.data, attrs))]

    def update(self, value):
        arr = self.transpose.get_x(self.data)
        attrs = self.get_enhanced_attrs(arr)
        self._texts[0].set_text(self.replace(value, self.data,
                                             attrs))

    def _swap_labels(self):
        plotter = self.plotter
        self.transpose._swap_labels()
        old_xlabel = self.value
        with plotter.no_validation:
            plotter[self.key] = self.ylabel.value
            plotter[self.ylabel.key] = old_xlabel


class Ylabel(TextBase, Formatoption):
    """
    Set the y-axis label

    Set the label for the y-axis.
    %(replace_note)s

    Possible types
    --------------
    str
        The text for the :func:`~matplotlib.pyplot.ylabel` function.

    See Also
    --------
    ylabelsize, ylabelweight, ylabelprops"""

    children = ['transpose']

    name = 'y-axis label'

    def initialize_plot(self, value):
        arr = self.transpose.get_y(self.data)
        attrs = self.get_enhanced_attrs(arr)
        self._texts = [self.ax.set_ylabel(self.replace(
            value, self.data, attrs))]

    def update(self, value):
        arr = self.transpose.get_y(self.data)
        attrs = self.get_enhanced_attrs(arr)
        self._texts[0].set_text(self.replace(
            value, self.data, attrs))


@docstrings.get_sectionsf('LabelOptions')
class LabelOptions(DictFormatoption):
    """
    Base formatoption class for label sizes

    Possible types
    --------------
    dict
        A dictionary with the keys ``'x'`` and (or) ``'y'`` to specify
        which ticks are managed. If the given value is not a dictionary with
        those keys, it is used for the x- and y-axis.
        The values in the dictionary can be one types below.
    """

    children = ['xlabel', 'ylabel']

    def update(self, value):
        for axis, val in value.items():
            self._text = getattr(self, axis + 'label')._texts[0]
            self.axis_str = axis
            self.update_axis(val)

    @abstractmethod
    def update_axis(self, value):
        pass


class LabelSize(LabelOptions):
    """
    Set the size of both, x- and y-label

    Possible types
    --------------
    %(LabelOptions.possible_types)s
    %(fontsizes)s

    See Also
    --------
    xlabel, ylabel, labelweight, labelprops"""

    group = 'labels'

    parents = ['labelprops']

    name = 'font size of x- and y-axis label'

    def update_axis(self, value):
        self._text.set_size(value)


class LabelWeight(LabelOptions):
    """
    Set the font size of both, x- and y-label

    Possible types
    --------------
    %(LabelOptions.possible_types)s
    %(fontweights)s

    See Also
    --------
    xlabel, ylabel, labelsize, labelprops"""

    group = 'labels'

    parents = ['labelprops']

    name = 'font weight of x- and y-axis label'

    def update_axis(self, value):
        self._text.set_weight(value)


class LabelProps(LabelOptions):
    """
    Set the font properties of both, x- and y-label

    Possible types
    --------------
    %(LabelOptions.possible_types)s
    dict
        Items may be any valid text property

    See Also
    --------
    xlabel, ylabel, labelsize, labelweight"""

    group = 'labels'

    children = ['xlabel', 'ylabel', 'labelsize', 'labelweight']

    name = 'font properties of x- and y-axis label'

    def update_axis(self, fontprops):
        fontprops = fontprops.copy()
        if 'size' not in fontprops and 'fontsize' not in fontprops:
            fontprops['size'] = self.labelsize.value[self.axis_str]
        if 'weight' not in fontprops and 'fontweight' not in fontprops:
            fontprops['weight'] = self.labelweight.value[self.axis_str]
        self._text.update(fontprops)


class Transpose(Formatoption):
    """
    Switch x- and y-axes

    By default, one-dimensional arrays have the dimension on the x-axis and two
    dimensional arrays have the first dimension on the y and the second on the
    x-axis. You can set this formatoption to True to change this behaviour

    Possible types
    --------------
    bool
        If True, axes are switched"""

    group = 'axes'

    name = 'Switch x- and y-axes'

    priority = BEFOREPLOTTING

    def __init__(self, *args, **kwargs):
        super(Transpose, self).__init__(*args, **kwargs)
        self.swap_funcs = {
            'ticks': self._swap_ticks,
            'ticklabels': self._swap_ticklabels,
            'limits': self._swap_limits,
            'labels': self._swap_labels,
            }

    def initialize_plot(self, value):
        pass

    def update(self, value):
        for func in six.itervalues(self.swap_funcs):
            func()

    def _swap_ticks(self):
        xaxis = self.ax.xaxis
        yaxis = self.ax.yaxis
        # swap major ticks
        old_xlocator = xaxis.get_major_locator()
        xaxis.set_major_locator(yaxis.get_major_locator())
        yaxis.set_major_locator(old_xlocator)
        # swap minor ticks
        old_xlocator = xaxis.get_minor_locator()
        xaxis.set_minor_locator(yaxis.get_minor_locator())
        yaxis.set_minor_locator(old_xlocator)

    def _swap_ticklabels(self):
        xaxis = self.ax.xaxis
        yaxis = self.ax.yaxis
        # swap major ticklabels
        old_xformatter = xaxis.get_major_formatter()
        xaxis.set_major_formatter(yaxis.get_major_formatter())
        yaxis.set_major_formatter(old_xformatter)
        # swap minor ticklabels
        old_xformatter = xaxis.get_minor_formatter()
        xaxis.set_minor_formatter(yaxis.get_minor_formatter())
        yaxis.set_minor_formatter(old_xformatter)

    def _swap_limits(self):
        old_xlim = list(self.ax.get_xlim())
        self.ax.set_xlim(*self.ax.get_ylim())
        self.ax.set_ylim(*old_xlim)

    def _swap_labels(self):
        old_xlabel = self.ax.get_xlabel()
        self.ax.set_xlabel(self.ax.get_ylabel())
        self.ax.set_ylabel(old_xlabel)

    def get_x(self, arr):
        if not hasattr(arr, 'ndim'):  # if the data object is an array list
            arr = arr[0]
        is_unstructured = self.decoder.is_unstructured(arr)
        if not is_unstructured and arr.ndim == 1:
            if self.value:
                return arr
            else:
                #: The x-coordinate name of the variable as stored in the
                #: dataset (might differ from the one in this array because
                #: this could also be time, z, y, etc.)
                ds_coord = arr.decoder.get_xname(
                    next(arr.iter_base_variables))
                xname = arr.dims[0]
        else:
            if self.value:
                ds_coord = arr.decoder.get_yname(
                    next(arr.iter_base_variables))
                xname = arr.dims[-2 if not is_unstructured else -1]
            else:
                ds_coord = arr.decoder.get_xname(
                    next(arr.iter_base_variables))
                xname = arr.dims[-1]
        if xname == ds_coord:
            if self.value:
                return arr.decoder.get_y(arr)
            return arr.decoder.get_x(arr)
        else:
            return arr.coords[xname]

    def get_y(self, arr):
        if not hasattr(arr, 'ndim'):  # if the data object is an array list
            arr = arr[0]
        is_unstructured = self.decoder.is_unstructured(arr)
        if not is_unstructured and arr.ndim == 1:
            if not self.value:
                return arr
            else:
                #: The x-coordinate name of the variable as stored in the
                #: dataset (might differ from the one in this array because
                #: this could also be time, z, y, etc.)
                ds_coord = arr.decoder.get_xname(
                    next(arr.iter_base_variables))
                yname = arr.dims[0]
        else:
            if not self.value:
                ds_coord = arr.decoder.get_yname(
                    next(arr.iter_base_variables))
                yname = arr.dims[-2 if not is_unstructured else -1]
            else:
                ds_coord = arr.decoder.get_xname(
                    next(arr.iter_base_variables))
                yname = arr.dims[-1]
        if yname == ds_coord:
            if self.value:
                return arr.decoder.get_x(arr)
            return arr.decoder.get_y(arr)
        else:
            return arr.coords[yname]


class LineColors(Formatoption):
    """
    Set the color coding

    This formatoptions sets the color of the lines, bars, etc.

    Possible types
    --------------
    None
        to use the axes color_cycle
    iterable
        (e.g. list) to specify the colors manually
    str
        %(cmap_note)s
    matplotlib.colors.ColorMap
        to automatically choose the colors according to the number of lines,
        etc. from the given colormap"""

    group = 'colors'

    priority = BEFOREPLOTTING

    name = 'Color cycle'

    def __init__(self, *args, **kwargs):
        super(LineColors, self).__init__(*args, **kwargs)
        self.colors = []
        self.default_colors = None

    def update(self, value):
        changed = self.plotter.has_changed(self.key)
        if value is None:
            ax = self.plotter.ax
            if float(mpl.__version__[:3]) < 1.5:
                self.color_cycle = ax._get_lines.color_cycle
            else:
                self.color_cycle = (props['color'] for props in
                                    ax._get_lines.prop_cycler)
            # use the default colors if it is reset to None
            if self.default_colors is not None:
                self.color_cycle = chain(self.default_colors, self.color_cycle)
        else:
            try:
                self.color_cycle = iter(get_cmap(value)(
                    np.linspace(0., 1., len(self.data),
                                endpoint=True)))
            except (ValueError, TypeError, KeyError):
                self.color_cycle = iter(value)
        if changed:
            self.colors = [
                next(self.color_cycle) for arr in self.data]
        else:  # then it is replotted
            # append new colors from color_cycle (if necessary)
            self.colors += [next(self.color_cycle) for _ in range(
                len(self.data) - len(self.colors))]
        # store the default colors
        if value is None and self.default_colors is None:
            self.default_colors = self.colors


class LinePlot(Formatoption):
    """
    Choose the line style of the plot

    Possible types
    --------------
    None
        Don't make any plotting
    str or list of str
        The line style  string to use (['solid' | 'dashed', 'dashdot', 'dotted'
        | (offset, on-off-dash-seq) | '-' | '--' | '-.' | ':' | 'None' | ' ' |
        ''])
    """

    plot_fmt = True

    group = 'plot'

    priority = BEFOREPLOTTING

    children = ['color', 'transpose']

    name = 'Line plot type'

    def __init__(self, *args, **kwargs):
        Formatoption.__init__(self, *args, **kwargs)
        self._kwargs = {}

    def update(self, value):
        # the real plot making is done by make_plot
        pass

    def make_plot(self):
        if hasattr(self, '_plot'):
            self.remove()
        if self.value is not None:
            self._plot = list(chain(*starmap(self.plot_arr, zip(
                self.data, self.color.colors, cycle(safe_list(self.value))))))

    def plot_arr(self, arr, c, ls):
        # since date time objects are covered better by pandas,
        # we convert to a series
        df = arr.to_series()
        if self.transpose.value:
            return self.ax.plot(df.values, df.index.values,
                                color=c, linestyle=ls, **self._kwargs)
        else:
            return self.ax.plot(df.index.values, df.values,
                                color=c, linestyle=ls, **self._kwargs)

    def remove(self):
        for artist in self._plot:
            artist.remove()
        del self._plot


class BarPlot(Formatoption):
    """
    Choose how to make the bar plot

    Possible types
    --------------
    None
        Don't make any plotting
    'bar'
        Create a usual bar plot with the bars side-by-side
    'stacked'
        Create stacked plot
    """

    plot_fmt = True

    group = 'plot'

    priority = BEFOREPLOTTING

    children = ['color', 'transpose']

    name = 'Bar plot type'

    def __init__(self, *args, **kwargs):
        Formatoption.__init__(self, *args, **kwargs)
        self._kwargs = {}

    def update(self, value):
        # the real plot making is done by make_plot
        pass

    def remove(self):
        for artist in self._plot:
            artist.remove()
        del self._plot

    def make_plot(self):
        if hasattr(self, '_plot'):
            self.remove()
        if self.value is not None:
            df = self.data.to_dataframe()
            old_containers = self.ax.containers[:]
            colors = self.color.colors
            if self.transpose.value:
                df.plot(kind='barh', color=colors, ax=self.ax, rot=0,
                        legend=False, grid=False,
                        stacked=self.value == 'stacked', **self._kwargs)
            else:
                df.plot(kind='bar', color=colors, ax=self.ax, rot=0,
                        legend=False, grid=False,
                        stacked=self.value == 'stacked', **self._kwargs)
            self._plot = [container for container in self.ax.containers
                          if container not in old_containers]


class ViolinXTicks(XTicks):

    __doc__ = XTicks.__doc__

    @property
    def array(self):
        if not self.transpose.value:
            return np.array(list(range(len(self.data))))
        return super(ViolinXTicks, self).array


class ViolinYTicks(YTicks):

    __doc__ = YTicks.__doc__

    @property
    def array(self):
        if self.transpose.value:
            return np.array(list(range(len(self.data))))
        return super(ViolinYTicks, self).array


class ViolinXTickLabels(XTickLabels, TextBase):
    __doc__ = XTickLabels.__doc__

    data_dependent = True

    def update_axis(self, value):
        if self.transpose.value or value is None:
            return super(ViolinXTickLabels, self).update_axis(value)
        if isinstance(value, six.string_types):
            self.set_ticklabels([
                self.replace(value, arr, self.get_enhanced_attrs(
                    arr, replot=True)) for arr in self.data])
        else:
            self.set_ticklabels([
                self.replace(val, arr, self.get_enhanced_attrs(
                    arr, replot=True)) for val, arr in zip(value, self.data)])


class ViolinYTickLabels(YTickLabels, TextBase):
    __doc__ = XTickLabels.__doc__

    data_dependent = True

    def update_axis(self, value):
        if self.transpose.value or value is None:
            return super(ViolinYTickLabels, self).update_axis(value)
        if isinstance(value, six.string_types):
            self.set_ticklabels([
                self.replace(value, arr, self.get_enhanced_attrs(
                    arr, replot=True)) for arr in self.data])
        else:
            self.set_ticklabels([
                self.replace(val, arr, self.get_enhanced_attrs(
                    arr, replot=True)) for val, arr in zip(value, self.data)])


class ViolinPlot(Formatoption):
    """
    Choose how to make the violin plot

    Possible types
    --------------
    None or False
        Don't make any plotting
    bool
        If True, visualize the violins
    """

    plot_fmt = True

    group = 'plot'

    priority = BEFOREPLOTTING

    children = ['color', 'transpose']

    name = 'Violin plot type'

    def __init__(self, *args, **kwargs):
        Formatoption.__init__(self, *args, **kwargs)
        self._kwargs = {}

    def update(self, value):
        # the real plot making is done by make_plot
        pass

    def remove(self):
        for artist in self._plot:
            artist.remove()
        del self._plot

    def make_plot(self):
        if hasattr(self, '_plot'):
            self.remove()
        if self.value:
            from seaborn import violinplot
            df = self.data.to_dataframe()
            old_artists = self.ax.containers[:] + self.ax.lines[:] \
                + self.ax.collections[:]
            palette = self.color.colors
            violinplot(data=df, palette=palette, ax=self.ax,
                       orient='h' if self.transpose.value else 'v',
                       **self._kwargs)
            artists = self.ax.containers + self.ax.lines + self.ax.collections
            self._plot = [
                artist for artist in artists
                if artist not in old_artists]


@docstrings.get_sectionsf('LimitBase')
@dedent
class LimitBase(Formatoption):
    """
    Base class for x- and y-limits

    Possible types
    --------------
    None
        To not change the current limits
    str or list [str, str]
        Automatically determine the ticks corresponding to the data. The given
        string determines how the limits are calculated.
        A string can be one of the following:

        rounded
            Sets the minimum and maximum of the limits to the rounded data
            minimum or maximum. Limits are rounded to the next 0.5 value with
            to the difference between data max- and minimum. The minimum
            will always be lower or equal than the data minimum, the maximum
            will always be higher or equal than the data maximum.
        roundedsym
            Same as `rounded` above but the limits are chosen such that they
            are symmetric around zero
        minmax
            Uses the minimum and maximum
        sym
            Same as minmax but symmetric around zero
    tuple (xmin, xmax)
        `xmin` is the smaller value, `xmax` the larger. Any of those values can
        be None or one of the strings above to use the corresponding value here
    """

    group = 'axes'

    children = ['transpose']

    connections = ['plot']

    #: :class:`bool` controlling whether the axis limits have to be relimited
    #: using the :meth:`matplotlib.axes.Axes.relim` method
    relim = True

    @abstractproperty
    def axisname(self):
        """The axis name (either ``'x'`` or ``'y'``)"""
        pass

    @abstractmethod
    def get_data(self):
        """A method to get the data"""
        pass

    @abstractmethod
    def set_limit(self, min_val, max_val):
        """The method to set the minimum and maximum limit

        Parameters
        ----------
        min_val: float
            The value for the lower limit
        max_val: float
            The value for the upper limit"""
        pass

    def __init__(self, *args, **kwargs):
        super(LimitBase, self).__init__(*args, **kwargs)
        self._calc_funcs = {
            'rounded': self._round_min_max,
            'roundedsym': self._roundedsym_min_max,
            'minmax': self._min_max,
            'sym': self._sym_min_max}

    def _round_min_max(self, vmin, vmax):
        try:
            exp = np.floor(np.log10(abs(vmax - vmin)))
            larger = round_to_05([vmin, vmax], exp, mode='l')
            smaller = round_to_05([vmin, vmax], exp, mode='s')
        except TypeError:
            self.logger.debug("Failed to calculate rounded limits!",
                              exc_info=True)
            return vmin, vmax
        return min([larger[0], smaller[0]]), max([larger[1], smaller[1]])

    def _min_max(self, vmin, vmax):
        return vmin, vmax

    def _roundedsym_min_max(self, vmin, vmax):
        vmax = max(map(abs, self._round_min_max(vmin, vmax)))
        return -vmax, vmax

    def _sym_min_max(self, vmin, vmax):
        vmax = max(abs(vmin), abs(vmax))
        return -vmax, vmax

    def update(self, value):
        value = list(value)
        data = np.asarray(self.get_data())
        vmin = data.min()
        vmax = data.max()
        for key, func in self._calc_funcs.items():
            if key in value:
                minmax = func(vmin, vmax)
                for i, val in enumerate(value):
                    if val == key:
                        value[i] = minmax[i]
        self.logger.debug('Setting %s with %s', self.key, value)
        self.set_limit(*value)


class Xlim(LimitBase):
    """
    Set the x-axis limits

    Possible types
    --------------
    %(LimitBase.possible_types)s

    See Also
    --------
    ylim
    """

    children = LimitBase.children + ['ylim']

    dependencies = ['xticks']

    axisname = 'x'

    name = 'x-axis limits'

    def get_data(self):
        df = self.data.to_dataframe()
        if self.transpose.value:
            return df
        return df.index

    def set_limit(self, *args):
        self.ax.set_xlim(*args)

    def initialize_plot(self, value):
        super(Xlim, self).initialize_plot(value)
        self.transpose.swap_funcs['limits'] = self._swap_limits

    def _swap_limits(self):
        self.transpose._swap_limits()
        old_xlim = self.value
        with self.plotter.no_validation:
            self.plotter[self.key] = self.ylim.value
            self.plotter[self.ylim.key] = old_xlim


class Ylim(LimitBase):
    """
    Set the y-axis limits

    Possible types
    --------------
    %(LimitBase.possible_types)s

    See Also
    --------
    xlim
    """
    children = LimitBase.children + ['xlim']

    dependencies = ['yticks']

    axisname = 'y'

    name = 'y-axis limits'

    def get_data(self):
        df = self.data.to_dataframe()
        if self.transpose.value:
            return df.index
        return df

    def set_limit(self, *args):
        self.ax.set_ylim(*args)


class ViolinXlim(Xlim):
    # xlim class for ViolinPlotter
    __doc__ = Xlim.__doc__

    def get_data(self):
        if not self.transpose.value:
            return np.array(
                [-0.5, len(self.data) - 0.5])
        return super(ViolinXlim, self).get_data()

    def _round_min_max(self, *args, **kwargs):
        if not self.transpose.value:
            return self.get_data()
        return super(ViolinXlim, self)._round_min_max(*args, **kwargs)


class BarXlim(ViolinXlim):
    # xlim class for bar plotter
    __doc__ = Xlim.__doc__

    def get_data(self):
        if self.transpose.value and self.plot.value:
            df = self.data.to_dataframe()
            return np.array([min([0, df.values.min()]), df.sum(axis=1).max()])
        elif not self.transpose.value:
            return np.array(
                [-0.5, len(self.data.to_dataframe().index) - 0.5])
        return super(BarXlim, self).get_data()


class Xlim2D(Xlim):
    __doc__ = Xlim.__doc__

    def get_data(self):
        xcoord = self.transpose.get_x(self.data)
        func = 'get_x' if not self.transpose.value else 'get_y'
        if (self.decoder.is_triangular(self.data) and
                xcoord.name == getattr(self.decoder, func)(self.data).name):
            triangles = self.decoder.get_triangles(self.data)
            if self.transpose.value:
                return triangles.y[triangles.triangles].ravel()
            else:
                return triangles.x[triangles.triangles].ravel()
        return self.decoder.get_plotbounds(xcoord)


class Ylim2D(Ylim):
    __doc__ = Ylim.__doc__

    def get_data(self):
        ycoord = self.transpose.get_y(self.data)
        func = 'get_x' if self.transpose.value else 'get_y'
        if (self.decoder.is_triangular(self.data) and
                ycoord.name == getattr(self.decoder, func)(self.data).name):
            triangles = self.decoder.get_triangles(self.data)
            if self.transpose.value:
                return triangles.x[triangles.triangles].ravel()
            else:
                return triangles.y[triangles.triangles].ravel()
        return self.decoder.get_plotbounds(self.transpose.get_y(self.data))


class ViolinYlim(Ylim):
    # Ylim class for ViolinPlotter
    __doc__ = Ylim.__doc__

    def get_data(self):
        if self.transpose.value:
            return np.array(
                [-0.5, len(self.data) - 0.5])
        return super(ViolinYlim, self).get_data()

    def _round_min_max(self, *args, **kwargs):
        if self.transpose.value:
            return self.get_data()
        return super(ViolinYlim, self)._round_min_max(*args, **kwargs)


class BarYlim(ViolinYlim):
    # ylim class for bar plotter
    __doc__ = Ylim.__doc__

    def get_data(self):
        if not self.transpose.value and self.plot.value:
            df = self.data.to_dataframe()
            return np.array([min(0, df.values.min()), df.sum(axis=1).max()])
        elif self.transpose.value:
            return np.array(
                [-0.5, len(self.data.to_dataframe().index) - 0.5])
        return super(BarYlim, self).get_data()


class XRotation(Formatoption):
    """
    Rotate the x-axis ticks

    Possible types
    --------------
    float
        The rotation angle in degrees

    See Also
    --------
    yrotation"""

    group = 'ticks'

    children = ['yticklabels']

    name = 'Rotate x-ticklabels'

    def update(self, value):
        for text in self.ax.get_xticklabels(which='both'):
            text.set_rotation(value)


class YRotation(Formatoption):
    """
    Rotate the y-axis ticks

    Possible types
    --------------
    float
        The rotation angle in degrees

    See Also
    --------
    xrotation"""

    group = 'ticks'

    children = ['yticklabels']

    name = 'Rotate y-ticklabels'

    def update(self, value):
        for text in self.ax.get_yticklabels(which='both'):
            text.set_rotation(value)


class CMap(Formatoption):
    """
    Specify the color map

    This formatoption specifies the color coding of the data via a
    :class:`matplotlib.colors.Colormap`

    Possible types
    --------------
    str
        %(cmap_note)s
    matplotlib.colors.Colormap
        The colormap instance to use

    See Also
    --------
    bounds: specifies the boundaries of the colormap"""

    group = 'colors'

    priority = BEFOREPLOTTING

    name = 'Colormap'

    def update(self, value):
        pass  # the colormap is set when plotting


class MissColor(Formatoption):
    """
    Set the color for missing values

    Possible types
    --------------
    None
        Use the default from the colormap
    string, tuple.
        Defines the color of the grid."""

    group = 'colors'

    priority = END

    dependencies = ['plot']

    connections = ['transform']

    name = 'Color of missing values'

    update_after_plot = True

    @property
    def triangles(self):
        """The :class:`matplotlib.tri.Triangulation` instance containing the
        spatial informations"""
        decoder = self.raw_data.decoder
        return decoder.get_triangles(self.data, self.data.coords, copy=True,
                                     nans='only')

    def update(self, value):
        if self.plotter.replot:
            self.remove()
        if not self.decoder.is_unstructured(self.data):
            mappable = self.plot.mappable
            if value is not None:
                mappable.get_cmap().set_bad(value)
            else:
                mappable.get_cmap().set_bad(alpha=0)
            mappable.changed()
        else:
            # The miss_color for triangular plots does not work here,
            # therefore we have to do a little workaround by making a plot
            # filled with 0 and the necessary colormap
            from matplotlib.tri import TriAnalyzer
            if value is None:
                return
            cmap = mcol.LinearSegmentedColormap.from_list(
                'dummy', [value, value], N=2)
            if hasattr(self, '_miss_color_plot'):
                self._miss_color_plot.set_cmap(cmap)
            else:
                triangles = self.triangles
                mratio = rcParams['plotter.maps.plot.min_circle_ratio']
                if mratio:
                    triangles.set_mask(
                        TriAnalyzer(triangles).get_flat_tri_mask(mratio))
                if self.transform is not None:
                    transform = self.transform.projection
                else:
                    transform = None
                self._miss_color_plot = self.ax.tripcolor(
                    triangles, np.zeros(len(triangles.triangles)),
                    rasterized=True, cmap=cmap, transform=transform,
                    zorder=self.plot.mappable.zorder - 0.2, facecolor=value)

    def remove(self):
        if hasattr(self, '_miss_color_plot'):
            try:
                self._miss_color_plot.remove()
                del self._miss_color_plot
            except ValueError:
                pass


@docstrings.get_sectionsf('Bounds', sections=['Possible types', 'Examples',
                                              'See Also'])
class Bounds(DataTicksCalculator):
    """
    Specify the boundaries of the colorbar

    Possible types
    --------------
    None
        make no normalization
    %(DataTicksCalculator.possible_types)s
    int
        Specifies how many ticks to use with the ``'rounded'`` option. I.e. if
        integer ``i``, then this is the same as ``['rounded', i]``.

    Examples
    --------
    Plot 11 bounds over the whole data range::

    >>> plotter.update(bounds='rounded')

    Plot 7 ticks over the whole data range where the maximal and minimal
    tick matches the data maximum and minimum::

    >>> plotter.update(bounds=['minmax', 7])

    Plot logarithmic bounds::

    >>> from matplotlib.colors import LogNorm
    >>> plotter.update(bounds=LogNorm())


    See Also
    --------
    cmap: Specifies the colormap"""

    group = 'colors'

    priority = BEFOREPLOTTING

    name = 'Boundaries of the color map'

    @property
    def value2share(self):
        """The normalization instance"""
        if len(self.bounds) > 1:
            return list(self.bounds)
        return self.norm

    def update(self, value):
        if value is None or isinstance(value, mpl.colors.Normalize):
            self.norm = value
            self.bounds = [0]
        else:
            if isinstance(value[0], six.string_types):
                value = self.calc_funcs[value[0]](*value[1:])
            self.bounds = value
            self.norm = mpl.colors.BoundaryNorm(
                value, len(value) - 1)


def _infer_interval_breaks(coord):
    """
    >>> _infer_interval_breaks(np.arange(5))
    array([-0.5,  0.5,  1.5,  2.5,  3.5,  4.5])
    """
    coord = np.asarray(coord)
    deltas = 0.5 * (coord[1:] - coord[:-1])
    first = coord[0] - deltas[0]
    last = coord[-1] + deltas[-1]
    return np.r_[[first], coord[:-1] + deltas, [last]]


@docstrings.get_sectionsf('Plot2D')
class Plot2D(Formatoption):
    """
    Choose how to visualize a 2-dimensional scalar data field

    Possible types
    --------------
    None
        Don't make any plotting
    'mesh'
        Use the :func:`matplotlib.pyplot.pcolormesh` function to make the plot
        or the :func:`matplotlib.pyplot.tripcolor` for an unstructered grid
    'tri'
        Use the :func:`matplotlib.pyplot.tripcolor` function to plot data on a
        triangular grid
    """

    plot_fmt = True

    group = 'plot'

    priority = BEFOREPLOTTING

    name = '2D plot type'

    children = ['cmap', 'bounds']

    @property
    def array(self):
        """The (masked) data array that is plotted"""
        arr = self.data.values
        return np.ma.masked_array(arr, mask=np.isnan(arr))

    @property
    def xbounds(self):
        """Boundaries of the x-coordinate"""
        raw_data = self.raw_data
        data = self.data
        coord = raw_data.decoder.get_x(data, coords=data.coords)
        return raw_data.decoder.get_plotbounds(coord)

    @property
    def ybounds(self):
        """Boundaries of the y-coordinate"""
        raw_data = self.raw_data
        data = self.data
        coord = raw_data.decoder.get_y(data, coords=data.coords)
        return raw_data.decoder.get_plotbounds(coord)

    @property
    def triangles(self):
        """The :class:`matplotlib.tri.Triangulation` instance containing the
        spatial informations"""
        decoder = self.raw_data.decoder
        return decoder.get_triangles(self.data, self.data.coords, copy=True,
                                     nans='skip')

    @property
    def mappable(self):
        """Returns the mappable that can be used for colorbars"""
        return self._plot

    def __init__(self, *args, **kwargs):
        Formatoption.__init__(self, *args, **kwargs)
        self._plot_funcs = {
            'mesh': self._pcolormesh,
            'tri': self._tripcolor}
        self._kwargs = {}

    def update(self, value):
        # the real plot making is done by make_plot
        pass

    def make_plot(self):
        # remove the plot if it shall be replotted or any of the dependencies
        # changed
        if self.plotter.replot or any(
                self.plotter.has_changed(key) for key in chain(
                    self.connections, self.dependencies, [self.key])):
            self.remove()
        if self.value is not None:
            self._plot_funcs[self.value]()

    def _pcolormesh(self):
        if self.raw_data.decoder.is_triangular(self.raw_data):
            return self._tripcolor()
        arr = self.array
        N = len(np.unique(self.bounds.norm(arr.ravel())))
        cmap = get_cmap(self.cmap.value, N)
        if hasattr(self, '_plot'):
            self._plot.update(dict(cmap=cmap, norm=self.bounds.norm))
            # for cartopy, we have to consider the wrapped collection if the
            # data has to be transformed
            try:
                coll = self._plot._wrapped_collection_fix
            except AttributeError:
                pass
            else:
                coll.update(dict(cmap=cmap, norm=self.bounds.norm))
        else:
            self._plot = self.ax.pcolormesh(
                self.xbounds, self.ybounds, arr, norm=self.bounds.norm,
                cmap=cmap, rasterized=True, **self._kwargs)

    def _tripcolor(self):
        from matplotlib.tri import TriAnalyzer
        triangles = self.triangles
        mratio = rcParams['plotter.maps.plot.min_circle_ratio']
        if mratio:
            triangles.set_mask(
                TriAnalyzer(triangles).get_flat_tri_mask(mratio))
        cmap = get_cmap(self.cmap.value, len(self.bounds.bounds) - 1 or None)
        if hasattr(self, '_plot'):
            self._plot.update(dict(cmap=cmap, norm=self.bounds.norm))
        else:
            arr = self.array
            self._plot = self.ax.tripcolor(
                triangles, arr[~np.isnan(arr)], norm=self.bounds.norm, cmap=cmap,
                rasterized=True, **self._kwargs)

    def remove(self):
        if hasattr(self, '_plot'):
            self._plot.remove()
            del self._plot


class DataGrid(Formatoption):
    """
    Show the grid of the data

    This formatoption shows the grid of the data (without labels)

    Possible types
    --------------
    None
        Don't show the data grid
    str
        A linestyle in the form ``'k-'``, where ``'k'`` is the color and
        ``'-'`` the linestyle.
    dict
        any keyword arguments that are passed to the plotting function (
        :func:`matplotlib.pyplot.triplot` for triangular grids and
        :func:`matplotlib.pyplot.hlines` for rectilinear grids)

    See Also
    --------
    psyplot.plotter.maps.FieldPlotter.xgrid
    psyplot.plotter.maps.FieldPlotter.ygrid"""

    children = ['transform']

    name = 'Grid of the data'

    @property
    def array(self):
        """The (masked) data array that is plotted"""
        arr = self.data.values
        return np.ma.masked_array(arr, mask=np.isnan(arr))

    @property
    def xbounds(self):
        """Boundaries of the x-coordinate"""
        raw_data = self.raw_data
        data = self.data
        coord = raw_data.decoder.get_x(data, coords=data.coords)
        return raw_data.decoder.get_plotbounds(coord)

    @property
    def ybounds(self):
        """Boundaries of the y-coordinate"""
        raw_data = self.raw_data
        data = self.data
        coord = raw_data.decoder.get_y(data, coords=data.coords)
        return raw_data.decoder.get_plotbounds(coord)

    @property
    def triangles(self):
        """The :class:`matplotlib.tri.Triangulation` instance containing the
        spatial informations"""
        decoder = self.raw_data.decoder
        return decoder.get_triangles(self.data, self.data.coords, copy=True)

    def _triplot(self, value):
        if isinstance(value, dict):
            self._artists = self.ax.triplot(self.triangles, **value)
        else:
            self._artists = self.ax.triplot(self.triangles, value)

    def _rectilinear_plot(self, value):
        if not isinstance(value, dict):
            value = dict(zip(
                ['linestyle', 'marker', 'color'],
                matplotlib.axes._base._process_plot_format(value)))
            del value['marker']
        ybounds = self.ybounds
        xbounds = self.xbounds
        if xbounds.ndim == 2:
            warn('The visualization of a datagrid is not implemented for '
                 'circumpolar grids', logger=self.logger)
        else:
            try:
                value.setdefault('transform', self.transform.projection)
            except AttributeError:
                pass
            self._artists = [
                self.ax.hlines(ybounds, xbounds.min(), xbounds.max(), **value),
                self.ax.vlines(xbounds, ybounds.min(), ybounds.max(), **value)]

    def update(self, value):
        if value is None:
            self.remove()
        else:
            if self.raw_data.decoder.is_triangular(self.raw_data):
                self._triplot(value)
            else:
                self._rectilinear_plot(value)

    def remove(self):
        if not hasattr(self, '_artists'):
            return
        for artist in self._artists:
            artist.remove()
        del self._artists


class SimplePlot2D(Plot2D):
    """
    Specify the plotting method

    Possible types
    --------------
    None
        Don't make any plotting
    'mesh'
        Use the :func:`matplotlib.pyplot.pcolormesh` function to make the plot
    """

    dependencies = Plot2D.dependencies + ['transpose']

    @property
    def array(self):
        if self.transpose.value:
            return super(SimplePlot2D, self).array.T
        else:
            return super(SimplePlot2D, self).array

    @property
    def triangles(self):
        triangles = super(SimplePlot2D, self).triangles
        if self.transpose.value:
            triangles.x, triangles.y = triangles.y, triangles.x
        return triangles

    @property
    def xbounds(self):
        return self.raw_data.decoder.get_plotbounds(self.transpose.get_x(
            self.data))

    @property
    def ybounds(self):
        return self.raw_data.decoder.get_plotbounds(self.transpose.get_y(
            self.data))


class XTicks2D(XTicks):

    __doc__ = XTicks.__doc__

    @property
    def data(self):
        da = super(XTicks, self).data
        if self.transpose.value:
            return da.coords[da.dims[-2]]
        else:
            return da.coords[da.dims[-1]]


class YTicks2D(YTicks):

    __doc__ = YTicks.__doc__

    @property
    def data(self):
        da = super(YTicks, self).data
        if self.transpose.value:
            return da.coords[da.dims[-1]]
        else:
            return da.coords[da.dims[-2]]


class Extend(Formatoption):
    """
    Draw arrows at the side of the colorbar

    Possible types
    --------------
    str {'neither', 'both', 'min' or 'max'}
        If not 'neither', make pointed end(s) for out-of-range values
    """

    group = 'colors'

    name = 'Ends of the colorbar'

    def update(self, value):
        # nothing to do here because the extend is set by the Cbar formatoption
        pass


class CbarSpacing(Formatoption):
    """
    Specify the spacing of the bounds in the colorbar

    Possible types
    --------------
    str {'uniform', 'proportional'}
        if ``'uniform'``, every color has exactly the same width in the
        colorbar, if ``'proportional'``, the size is chosen according to the
        data"""

    group = 'colors'

    connections = ['cbar']

    name = 'Spacing of the colorbar'

    def update(self, value):
        self.cbar._kwargs['spacing'] = value


@docstrings.get_sectionsf('Cbar')
class Cbar(Formatoption):
    """
    Specify the position of the colorbars

    Possible types
    --------------
    bool
        True: defaults to 'b'
        False: Don't draw any colorbar
    str
        The string can be a combination of one of the following strings:
        {'fr', 'fb', 'fl', 'ft', 'b', 'r', 'sv', 'sh'}

        - 'b', 'r' stand for bottom and right of the axes
        - 'fr', 'fb', 'fl', 'ft' stand for bottom, right, left and top of the
          figure
        - 'sv' and 'sh' stand for a vertical or horizontal colorbar in a
          separate figure
    list
        A containing one of the above positions

    Examples
    --------
    Draw a colorbar at the bottom and left of the axes::

    >>> plotter.update(cbar='bl')"""

    dependencies = ['plot', 'cmap', 'bounds', 'extend', 'cbarspacing']

    group = 'colors'

    name = 'Position of the colorbar'

    priority = END + 0.1

    figure_positions = {'fr', 'fb', 'fl', 'ft', 'b', 'r', 'l', 't'}

    original_position = None

    @property
    def init_kwargs(self):
        return dict(chain(six.iteritems(super(Cbar, self).init_kwargs),
                          [('other_cbars', self.other_cbars)]))

    @docstrings.dedent
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        %(Formatoption.parameters)s
        other_cbars: list of str
            List of other colorbar formatoption keys (necessary for a
            sufficient resizing of the axes)"""
        self.other_cbars = kwargs.pop('other_cbars', [])
        super(Cbar, self).__init__(*args, **kwargs)
        self._kwargs = {}

    def initialize_plot(self, value):
        self._set_original_position()
        self.cbars = {}
        super(Cbar, self).initialize_plot(value)

    def _set_original_position(self):
        """Gets and sets the original position of the axes without colorbar"""
        # this is somewhat a hack to make sure that we get the right position
        # although the figure has not been drawn so far
        for key in self.other_cbars:
            fmto = getattr(self.plotter, key)
            if fmto.original_position:
                self.original_position = fmto.original_position
                return
        ax = self.ax
        if ax._adjustable in ['box', 'box-forced']:
            figW, figH = ax.get_figure().get_size_inches()
            fig_aspect = figH / figW
            position = ax.get_position(True)
            pb = position.frozen()
            box_aspect = ax.get_data_ratio()
            pb1 = pb.shrunk_to_aspect(box_aspect, pb, fig_aspect)
            self.original_position = pb1.anchored(ax.get_anchor(), pb)
        else:
            self.original_position = ax.get_position(True)

    @property
    def value2share(self):
        """Those colorbar positions that are directly at the axes"""
        return self.value.intersection(['r', 'b', 'l', 't'])

    def update(self, value):
        """
        Updates the colorbar

        Parameters
        ----------
        value
            The value to update (see possible types)
        no_fig_cbars
            Does not update the colorbars that are not in the axes of this
            plot"""
        plotter = self.plotter
        if plotter.replot or any(
                plotter.has_changed(key, False) for key in self.dependencies
                if getattr(self, key, None) is not None and key not in [
                        self._child_mapping['cmap'], self._child_mapping[
                            'bounds']]):
            cbars2delete = set(self.cbars)
        else:
            cbars2delete = set(self.cbars).difference(value)
        if cbars2delete:
            # if the colorbars are in the figure of the axes, we have to first
            # remove all the colorbars and then redraw it in order to make
            # sure that the axes gets the right position
            if cbars2delete & self.figure_positions:
                cbars2delete.update(self.figure_positions)
                self.remove(positions=cbars2delete)
                # remove other cbars
                for key in self.other_cbars:
                    fmto = getattr(plotter, key)
                    fmto.remove(self.figure_positions)
                # redraw other cbars
                for key in self.other_cbars:
                    fmto = getattr(plotter, key)
                    fmto.update(fmto.value)
            else:
                self.remove(positions=cbars2delete)
        if self.plot.value is None:
            return
        for pos in value.intersection(self.cbars):
            self.update_colorbar(pos)
        for pos in sorted(value.difference(self.cbars)):
            self.draw_colorbar(pos)
        plotter._figs2draw.update(map(lambda cbar: cbar.ax.get_figure(),
                                      six.itervalues(self.cbars)))

    def update_colorbar(self, pos):
        cbar = self.cbars[pos]
        cbar.set_norm(self.plot.mappable.norm)
        cbar.set_cmap(self.plot.mappable.cmap)
        cbar.draw_all()

    def remove(self, positions='all'):
        import matplotlib.pyplot as plt

        def try2remove(cbar):
            try:
                cbar.remove()
            except KeyError:
                # the colorbar has been removed already from some other
                # Cbar instance
                pass
        if positions == 'all':
            positions = self.cbars.keys()
        positions = set(positions).intersection(self.cbars.keys())
        if not positions:
            return
        adjustment = {}
        to_adjust = {'fr': 'right', 'fl': 'left', 'ft': 'top', 'fb': 'bottom'}
        for pos in positions:
            cbar = self.cbars.pop(pos)
            if pos in ['sh', 'sv']:
                plt.close(cbar.ax.get_figure())
            else:
                # set the axes for the mappable if this has been removed
                if cbar.mappable.axes is None:
                    cbar.mappable.axes = self.plotter.ax
                    try2remove(cbar)
                    cbar.mappable.axes = None
                else:
                    try2remove(cbar)
#                fig = cbar.ax.get_figure()
#                fig.delaxes(self.cbars[pos].ax)
                if pos in to_adjust:
                    adjustment[to_adjust[pos]] = mpl.rcParams[
                        'figure.subplot.' + to_adjust[pos]]
        if adjustment:
            self.ax.get_figure().subplots_adjust(**adjustment)
        if self.figure_positions.intersection(positions):
            self.ax.set_position(self.original_position)
        return

    def draw_colorbar(self, pos):
        import matplotlib.pyplot as plt
        # TODO: Manage to draw colorbars left and top (gridspec does not work)
        orientations = {
            # 'b': 'bottom', 'r': 'right', 'l': 'left', 't': 'top',
            'b': 'horizontal', 'r': 'vertical',
            'fr': 'vertical', 'fl': 'vertical', 'sv': 'vertical',
            'ft': 'horizontal', 'fb': 'horizontal', 'sh': 'horizontal'}

        orientation = orientations[pos]
        kwargs = self._kwargs.copy()
        if pos in ['b', 'r', 'l', 't']:
            fig = self.ax.get_figure()
            # kwargs = {'ax': self.ax, 'location': orientation}
            kwargs.update({'ax': self.ax, 'orientation': orientation})
        elif pos == 'sh':
            fig = plt.figure(figsize=(8, 1))
            kwargs.update({'cax': fig.add_axes([0.05, 0.5, 0.9, 0.3])})
            self.plotter._figs2draw.add(fig)  # add figure for drawing
        elif pos == 'sv':
            fig = plt.figure(figsize=(1, 8))
            kwargs.update({'cax': fig.add_axes([0.3, 0.05, 0.3, 0.9])})
            self.plotter._figs2draw.add(fig)  # add figure for drawing
        else:
            fig = self.ax.get_figure()
            if pos == 'fb':
                fig.subplots_adjust(bottom=0.2)
                kwargs['cax'] = fig.add_axes(
                    [0.125, 0.135, 0.775, 0.05],
                    label=self.raw_data.arr_name + '_fb')
            elif pos == 'fr':
                fig.subplots_adjust(right=0.8)
                kwargs['cax'] = fig.add_axes(
                    [0.825, 0.25, 0.035, 0.6],
                    label=self.raw_data.arr_name + '_fr')
            elif pos == 'fl':
                fig.subplots_adjust(left=0.225)
                kwargs['cax'] = fig.add_axes(
                    [0.075, 0.25, 0.035, 0.6],
                    label=self.raw_data.arr_name + '_fl')
            elif pos == 'ft':
                fig.subplots_adjust(top=0.75)
                kwargs['cax'] = fig.add_axes(
                    [0.125, 0.825, 0.775, 0.05],
                    label=self.raw_data.arr_name + '_ft')
        kwargs['extend'] = self.extend.value
        if 'location' not in kwargs:
            kwargs['orientation'] = orientation
        self.cbars[pos] = fig.colorbar(
            self.plot.mappable, **kwargs)
        if pos == 'fl':
            # draw tick labels left
            self.cbars[pos].ax.tick_params('y', labelleft=True,
                                           labelright=False)
        elif pos == 'ft':
            # draw ticklabels at the top
            self.cbars[pos].ax.tick_params('x', labeltop=True,
                                           labelbottom=False)


class CLabel(TextBase, Formatoption):
    """
    Show the colorbar label

    Set the label of the colorbar.
    %(replace_note)s

    Possible types
    --------------
    str
        The title for the :meth:`~matplotlib.colorbar.Colorbar.set_label`
        method.

    See Also
    --------
    clabelsize, clabelweight, clabelprops"""

    children = ['plot']

    dependencies = ['cbar']

    name = 'Colorbar label'

    data_dependent = True

    group = 'labels'

    axis_locations = {
            'b': 'x', 'r': 'y', 'l': 'y', 't': 'x',  # axes locations
            'fr': 'y', 'fl': 'y', 'sv': 'y',         # vertical figure cbars
            'ft': 'x', 'fb': 'x', 'sh': 'x'}         # horizontal figure cbars

    def update(self, value):
        arr = self.plot.data
        self.texts = []
        for pos, cbar in six.iteritems(self.cbar.cbars):
            cbar.set_label(self.replace(
                    value, arr, attrs=self.get_enhanced_attrs(
                        self.plot.data)))
            self.texts.append(getattr(
                cbar.ax, self.axis_locations[pos] + 'axis').get_label())
            if pos == 'fl':
                cbar.ax.yaxis.set_label_position('left')
            elif pos == 'ft':
                cbar.ax.xaxis.set_label_position('top')


class VCLabel(CLabel):
    """
    Show the colorbar label of the vector plot

    Set the label of the colorbar.
    %(replace_note)s

    Possible types
    --------------
    str
        The title for the :meth:`~matplotlib.colorbar.Colorbar.set_label`
        method.

    See Also
    --------
    vclabelsize, vclabelweight, vclabelprops"""
    pass


class CbarOptions(Formatoption):
    """Base class for colorbar formatoptions"""

    which = 'major'

    children = ['plot']

    dependencies = ['cbar']

    @property
    def colorbar(self):
        try:
            return self._colorbar
        except AttributeError:
            pos, cbar = next(six.iteritems(self.cbar.cbars))
            self.position = pos
            self.colorbar = cbar
            return self.colorbar

    @colorbar.setter
    def colorbar(self, cbar):
        self._colorbar = cbar

    @property
    def axis(self):
        """axis of the colorbar with the ticks. Will be overwritten during
        update process."""
        return getattr(
            self.colorbar.ax, self.axis_locations[self.position] + 'axis')

    @property
    def axisname(self):
        return self.axis_locations[self.position]

    @property
    def data(self):
        return self.plot.data

    axis_locations = CLabel.axis_locations

    def update(self, value):
        for pos, cbar in six.iteritems(self.cbar.cbars):
            self.colorbar = cbar
            self.position = pos
            self.update_axis(value)


@docstrings.get_sectionsf('CTicks')
class CTicks(CbarOptions, TicksBase):
    """
    Specify the tick locations of the colorbar

    Possible types
    --------------
    None
        use the default ticks
    %(DataTicksCalculator.possible_types)s
        bounds
            let the :attr:`bounds` keyword determine the ticks. An additional
            integer `i` may be specified to only use every i-th bound as a tick
            (see also `int` below)
    int
        Specifies how many ticks to use with the ``'bounds'`` option. I.e. if
        integer ``i``, then this is the same as ``['bounds', i]``.

    See Also
    --------
    cticklabels
    """

    dependencies = CbarOptions.dependencies + ['bounds']

    name = 'Colorbar ticks'

    @property
    def default_locator(self):
        """Default locator of the axis of the colorbars"""
        try:
            return self._default_locator
        except AttributeError:
            self.set_default_locators()
        return self._default_locator

    @default_locator.setter
    def default_locator(self, locator):
        self._default_locator = locator

    def __init__(self, *args, **kwargs):
        super(CTicks, self).__init__(*args, **kwargs)
        self.calc_funcs['bounds'] = self._bounds_ticks

    def set_ticks(self, value):
        self.colorbar.set_ticks(value)

    def _bounds_ticks(self, step=None):
        step = step or 1
        return self.bounds.bounds[::step]

    def update_axis(self, value):
        cbar = self.colorbar
        if value is None:
            cbar.locator = self.default_locator
            cbar.formatter = self.default_formatter
            cbar.update_ticks()
        else:
            TicksBase.update_axis(self, value)

    def set_default_locators(self, *args, **kwargs):
        if self.cbar.cbars:
            self.default_locator = self.colorbar.locator
            self.default_formatter = self.colorbar.formatter


class VectorCTicks(CTicks):
    """
    Specify the tick locations of the vector colorbar

    Possible types
    --------------
    %(CTicks.possible_types)s

    See Also
    --------
    cticklabels, vcticklabels
    """

    dependencies = CTicks.dependencies + ['color']

    @property
    def array(self):
        arr = self.color._color_array
        return arr[~np.isnan(arr)]


class CTickLabels(CbarOptions, TickLabelsBase):
    """
    Specify the colorbar ticklabels

    Possible types
    --------------
    %(TickLabelsBase.possible_types)s

    See Also
    --------
    cticks, cticksize, ctickweight, ctickprops
    vcticks, vcticksize, vctickweight, vctickprops
    """

    name = 'Colorbar ticklabels'

    @property
    def default_formatters(self):
        """Default locator of the axis of the colorbars"""
        if self._default_formatters:
            return self._default_formatters
        else:
            self.set_default_formatters()
        return self._default_formatters

    @default_formatters.setter
    def default_formatters(self, d):  # d is expected to be a dictionary
        self._default_formatters = d

    def set_default_formatters(self):
        if self.cbar.cbars:
            self.default_formatters = {self.which: self.colorbar.formatter}

    def set_formatter(self, formatter):
        cbar = self.colorbar
        cbar.formatter = formatter
        cbar.update_ticks()


class CTickSize(CbarOptions, TickSizeBase):
    """
    Specify the font size of the colorbar ticklabels

    Possible types
    --------------
    %(fontsizes)s

    See Also
    --------
    ctickweight, ctickprops, cticklabels, cticks
    vctickweight, vctickprops, vcticklabels, vcticks"""

    group = 'colors'

    name = 'Font size of the colorbar ticklabels'


class CTickWeight(CbarOptions, TickWeightBase):
    """
    Specify the fontweight of the colorbar ticklabels

    Possible types
    --------------
    %(fontweights)s

    See Also
    --------
    cticksize, ctickprops, cticklabels, cticks
    vcticksize, vctickprops, vcticklabels, vcticks"""

    group = 'colors'

    name = 'Font weight of the colorbar ticklabels'


class CTickProps(CbarOptions, TickPropsBase):
    """
    Specify the font properties of the colorbar ticklabels

    Possible types
    --------------
    %(TickPropsBase.possible_types)s

    See Also
    --------
    cticksize, ctickweight, cticklabels, cticks
    vcticksize, vctickweight, vcticklabels, vcticks"""

    children = CbarOptions.children + TickPropsBase.children

    group = 'colors'

    name = 'Font properties of the colorbar ticklabels'


class ArrowSize(Formatoption):
    """
    Change the size of the arrows

    Possible types
    --------------
    None
        make no scaling
    float
        Factor scaling the size of the arrows

    See Also
    --------
    arrowstyle, linewidth, density, color"""

    group = 'vector'

    priority = BEFOREPLOTTING

    dependencies = ['plot']

    name = 'Size of the arrows'

    def update(self, value):
        kwargs = self.plot._kwargs
        if self.plot.value == 'stream':
            kwargs.pop('scale', None)
            kwargs['arrowsize'] = value or 1.0
        else:
            kwargs.pop('arrowsize', None)
            kwargs['scale'] = value


class ArrowStyle(Formatoption):
    """Change the style of the arrows

    Possible types
    --------------
    str
        Any arrow style string (see
        :class:`~matplotlib.patches.FancyArrowPatch`)

    Notes
    -----
    This formatoption only has an effect for stream plots

    See Also
    --------
    arrowsize, linewidth, density, color"""

    group = 'vector'

    priority = BEFOREPLOTTING

    dependencies = ['plot']

    name = 'Style of the arrows'

    def update(self, value):
        if self.plot.value == 'stream':
            self.plot._kwargs['arrowstyle'] = value
        else:
            self.plot._kwargs.pop('arrowstyle', None)


@docstrings.get_sectionsf('WindCalculator')
class VectorCalculator(Formatoption):
    """
    Abstract formatoption that provides calculation functions for speed, etc.

    Possible types
    --------------
    string {'absolute', 'u', 'v'}
        Strings may define how the formatoption is calculated. Possible strings
        are

        - **absolute**: for the absolute wind speed
        - **u**: for the u component
        - **v**: for the v component
    """

    dependencies = ['plot']

    priority = BEFOREPLOTTING

    data_dependent = True

    def __init__(self, *args, **kwargs):
        super(VectorCalculator, self).__init__(*args, **kwargs)
        self._calc_funcs = {
            'absolute': self._calc_speed,
            'u': self._get_u,
            'v': self._get_v}

    def _maybe_ravel(self, arr):
        if self.plot.value == 'quiver':
            return np.ravel(arr)
        return np.asarray(arr)
        arr = np.asarray(arr)
        arr[np.isnan(arr)] = 0.0
        return arr

    def _calc_speed(self, scale=1.0):
        data = self.plot.data
        return self._maybe_ravel(
            np.sqrt(data[0].values**2 + data[1].values**2)) * scale

    def _get_u(self, scale=1.0):
        return self._maybe_ravel(self.plot.data[0].values) * scale

    def _get_v(self, scale=1.0):
        return self._maybe_ravel(self.plot.data[1].values) * scale


class VectorLineWidth(VectorCalculator):
    """
    Change the linewidth of the arrows

    Possible types
    --------------
    float
        give the linewidth explicitly
    %(WindCalculator.possible_types)s
    tuple (string, float)
        `string` may be one of the above strings, `float` may be a scaling
        factor
    2D-array
        The values determine the linewidth for each plotted arrow. Note that
        the shape has to match the one of u and v.

    See Also
    --------
    arrowsize, arrowstyle, density, color"""

    name = 'Linewidth of the arrows'

    def update(self, value):
        if value is None:
            self.plot._kwargs['linewidth'] = 0 if self.plot.value == 'quiver' \
                else None
        elif np.asarray(value).ndim and isinstance(value[0], six.string_types):
            self.plot._kwargs['linewidth'] = self._calc_funcs[value[0]](
                *value[1:])
        else:
            self.plot._kwargs['linewidth'] = self._maybe_ravel(value)


class VectorColor(VectorCalculator):
    """
    Set the color for the arrows

    This formatoption can be used to set a single color for the vectors or
    define the color coding

    Possible types
    --------------
    float
        Determines the greyness
    color
        Defines the same color for all arrows. The string can be either a html
        hex string (e.g. '#eeefff'), a single letter (e.g. 'b': blue,
        'g': green, 'r': red, 'c': cyan, 'm': magenta, 'y': yellow, 'k': black,
        'w': white) or any other color
    %(WindCalculator.possible_types)s
    2D-array
        The values determine the color for each plotted arrow. Note that
        the shape has to match the one of u and v.

    See Also
    --------
    arrowsize, arrowstyle, density, linewidth"""

    dependencies = VectorCalculator.dependencies + ['cmap', 'bounds']

    group = 'colors'

    name = 'Color of the arrows'

    def update(self, value):
        try:
            value = validate_color(value)
            self.colored = False
        except ValueError:
            if (isinstance(value, six.string_types) and
                    value in self._calc_funcs):
                value = self._calc_funcs[value]()
                self.colored = True
                self._color_array = value
            try:
                value = validate_float(value)
                self.colored = False
            except ValueError:
                value = self._maybe_ravel(value)
                self.colored = True
                self._color_array = value
        if self.plot.value == 'quiver' and self.colored:
            self.plot._args = [value]
            self.plot._kwargs.pop('color', None)
        else:
            self.plot._args = []
            self.plot._kwargs['color'] = value
        if self.colored:
            self._set_cmap()
        else:
            self._delete_cmap()

    def _set_cmap(self):
        if self.plotter.has_changed(self.key) or self.plotter._initializing:
            self.bounds.update(self.bounds.value)
        self.plot._kwargs['cmap'] = get_cmap(
            self.cmap.value, len(self.bounds.bounds) - 1 or None)
        self.plot._kwargs['norm'] = self.bounds.norm

    def _delete_cmap(self):
        self.plot._kwargs.pop('cmap', None)
        self.plot._kwargs.pop('norm', None)


@docstrings.get_sectionsf('Density')
class Density(Formatoption):
    """
    Change the density of the arrows

    Possible types
    --------------
    float
        Scales the density of the arrows in x- and y-direction (1.0 means
        no scaling)
    tuple (x, y)
        Defines the scaling in x- and y-direction manually

    Notes
    -----
    quiver plots do not support density scaling
    """

    dependencies = ['plot']

    group = 'vector'

    name = 'Density of the arrows'

    priority = BEFOREPLOTTING

    data_dependent = True

    def __init__(self, *args, **kwargs):
        super(Density, self).__init__(*args, **kwargs)
        self._density_funcs = {
            'stream': self._set_stream_density,
            'quiver': self._set_quiver_density}
        self._remove_funcs = {
            'stream': self._unset_stream_density,
            'quiver': self._unset_quiver_density}

    def update(self, value):
        has_changed = self.plotter.has_changed(self.plot.key)
        if has_changed:
            self.remove(has_changed[0])
        try:
            value = tuple(value)
        except TypeError:
            value = [value, value]
        if self.plot.value:
            self._density_funcs[self.plot.value](value)

    def _set_stream_density(self, value):
        return
        self.plot._kwargs['density'] = value

    def _set_quiver_density(self, value):
        if any(val != 1.0 for val in value):
            warn("Quiver plot does not support the density keyword!",
                 PsyPlotRuntimeWarning, logger=self.logger)

    def _unset_stream_density(self):
        self.plot._kwargs.pop('density', None)

    def _unset_quiver_density(self):
        pass

    def remove(self, plot_type=None):
        plot_type = plot_type or self.plot.value
        self._remove_funcs[plot_type]()


class VectorPlot(Formatoption):
    """
    Choose the vector plot type

    Possible types
    --------------
    str
        Plot types can be either

        quiver
            to make a quiver plot
        stream
            to make a stream plot"""

    plot_fmt = True

    group = 'plot'

    name = 'Plot type of the arrows'

    priority = BEFOREPLOTTING

    children = ['cmap', 'bounds']

    connections = ['transpose', 'transform', 'arrowsize', 'arrowstyle',
                   'density', 'linewidth', 'color']

    @property
    def mappable(self):
        """The mappable, i.e. the container of the plot"""
        if self.value == 'stream':
            return self._plot.lines
        else:
            return self._plot

    def __init__(self, *args, **kwargs):
        Formatoption.__init__(self, *args, **kwargs)
        self._plot_funcs = {
            'quiver': self._quiver_plot,
            'stream': self._stream_plot}
        self._args = []
        self._kwargs = {}

    def update(self, value):
        pass
        # the real plot making is done by make_plot but we store the value here
        # in case it is shared

    def make_plot(self):
        # remove the plot if it shall be replotted or any of the dependencies
        # changed. Otherwise there is nothing to change
        if hasattr(self, '_plot') and (self.plotter.replot or any(
                self.plotter.has_changed(key) for key in chain(
                    self.connections, self.dependencies, [self.key]))):
            self.remove()
        if not hasattr(self, "_plot") and self.value is not None:
            self._plot_funcs[self.value]()

    def _quiver_plot(self):
        x, y, u, v = self._get_data()
        self._plot = self.ax.quiver(x, y, u, v, *self._args, rasterized=True,
                                    **self._kwargs)

    def _stream_plot(self):
        x, y, u, v = self._get_data()
        self._plot = self.ax.streamplot(x, y, u, v, **self._kwargs)

    def _get_data(self):
        data = self.data
        if self.transpose.value:
            u = data[0].T.values
            v = data[1].T.values
        else:
            u, v = data.values
        x = self.transpose.get_x(data)
        y = self.transpose.get_y(data)
        return x, y, u, v

    def remove(self):
        def keep(x):
            return not isinstance(x, mpl.patches.FancyArrowPatch)
        if not hasattr(self, '_plot'):
            return
        if isinstance(self._plot, mpl.streamplot.StreamplotSet):
            try:
                self._plot.lines.remove()
            except ValueError:
                pass
            # remove arrows
            self.ax.patches = [patch for patch in self.ax.patches
                               if keep(patch)]
        else:
            try:
                self._plot.remove()
            except ValueError:  # the artist has already been removed
                pass
        del self._plot


class CombinedVectorPlot(VectorPlot):

    __doc__ = VectorPlot.__doc__

    def update(self, *args, **kwargs):
        self._kwargs['zorder'] = 2
        super(CombinedVectorPlot, self).update(*args, **kwargs)


class VectorCbar(Cbar):
    """
    Specify the position of the vector plot colorbars

    Possible types
    --------------
    %(Cbar.possible_types)s
    """

    dependencies = Cbar.dependencies + ['color']

    priority = END

    def update(self, *args, **kwargs):
        if self.color.colored:
            super(VectorCbar, self).update(*args, **kwargs)
        else:
            self.remove()


class VectorBounds(Bounds):
    """
    Specify the boundaries of the vector colorbar

    Possible types
    --------------
    %(Bounds.possible_types)s

    Examples
    --------
    %(Bounds.examples)s

    See Also
    --------
    %(Bounds.see_also)s"""

    parents = ['color']

    @property
    def array(self):
        arr = self.color._color_array
        return arr[~np.isnan(arr)]

    def update(self, *args, **kwargs):
        if not self.color.colored:
            return
        return super(VectorBounds, self).update(*args, **kwargs)


class LegendLabels(Formatoption, TextBase):
    """
    Set the labels of the arrays in the legend

    This formatoption specifies the labels for each array in the legend.
    %(replace_note)s

    Possible types
    --------------
    str:
        A single string that shall be used for all arrays.
    list of str:
        Same as a single string but specified for each array

    See Also
    --------
    legend"""

    data_dependent = True

    name = 'Labels in the legend'

    def update(self, value):
        if isinstance(value, six.string_types):
            self.labels = [
                self.replace(value, arr, self.get_enhanced_attrs(
                    arr, replot=True))
                for arr in self.data]
        else:
            self.labels = [
                self.replace(val, arr, self.get_enhanced_attrs(
                    arr, replot=True)) for val, arr in zip(value, self.data)]


class Legend(Formatoption):
    """
    Draw a legend

    This formatoption determines where and if to draw the legend. It uses the
    :attr:`labels` formatoption to determine the labels.

    Possible types
    --------------
    bool
        Draw a legend or not
    str or int
        Specifies where to plot the legend (i.e. the location)
    dict
        Give the keywords for the :func:`matplotlib.pyplot.legend` function

    See Also
    --------
    labels"""

    dependencies = ['legendlabels', 'plot']

    name = 'Properties of the legend'

    def update(self, value):
        labels = self.legendlabels.labels
        self.remove()
        if not value:
            return
        if value is True:
            value == 'best'
        if not isinstance(value, dict):
            value = {'loc': value}
        self.legend = self.ax.legend(labels, **value)

    def remove(self):
        if hasattr(self, 'legend'):
            self.legend.remove()


class XYTickPlotter(Plotter):
    """Plotter class for x- and y-ticks and x- and y- ticklabels
    """
    _rcparams_string = ['plotter.simple.']

    transpose = Transpose('transpose')
    xticks = XTicks('xticks')
    xticklabels = XTickLabels('xticklabels')
    yticks = YTicks('yticks')
    yticklabels = YTickLabels('yticklabels')
    ticksize = TickSize('ticksize')
    tickweight = TickWeight('tickweight')
    xtickprops = XTickProps('xtickprops')
    ytickprops = YTickProps('ytickprops')
    xlabel = Xlabel('xlabel')
    ylabel = Ylabel('ylabel')
    labelsize = LabelSize('labelsize')
    labelweight = LabelWeight('labelweight')
    labelprops = LabelProps('labelprops')
    xrotation = XRotation('xrotation')
    yrotation = YRotation('yrotation')


class Base2D(Plotter):
    """Base plotter for 2-dimensional plots
    """

    _rcparams_string = ['plotter.plot2d.']

    cmap = CMap('cmap')
    bounds = Bounds('bounds')
    extend = Extend('extend')
    cbar = Cbar('cbar')
    plot = None
    clabel = CLabel('clabel')
    clabelsize = label_size(clabel, 'Colorbar label', dependencies=['clabel'])
    clabelweight = label_weight(clabel, 'Colorbar label',
                                dependencies=['clabel'])
    cbarspacing = CbarSpacing('cbarspacing')
    clabelprops = label_props(clabel, 'Colorbar label',
                              dependencies=['clabel'])
    cticks = CTicks('cticks')
    cticklabels = CTickLabels('cticklabels')
    cticksize = CTickSize('cticksize')
    ctickweight = CTickWeight('ctickweight')
    ctickprops = CTickProps('ctickprops', ticksize='cticksize')
    datagrid = DataGrid('datagrid', index_in_list=0)


class LinePlotter(BasePlotter, XYTickPlotter):
    """Plotter for simple one-dimensional line plots
    """
    _rcparams_string = ['plotter.line.']

    transpose = Transpose('transpose')
    axiscolor = AxisColor('axiscolor')
    grid = Grid('grid')
    color = LineColors('color')
    plot = LinePlot('plot')
    xlim = Xlim('xlim')
    ylim = Ylim('ylim')
    legendlabels = LegendLabels('legendlabels')
    legend = Legend('legend')

    def _set_data(self, *args, **kwargs):
        Plotter._set_data(self, *args, **kwargs)
        data = self.plot_data
        if not isinstance(data, InteractiveList):
            self.plot_data = InteractiveList(
                [data], arr_name=data.arr_name, attrs=data.attrs)

    @classmethod
    @docstrings.dedent
    def check_data(cls, name, dims, is_unstructured=None):
        """
        A validation method for the data shape

        Parameters
        ----------
        name: str or list of str
            The variable names (one variable per array)
        dims: list with length 1 or list of lists with length 1
            The dimension of the arrays. Only 1D-Arrays are allowed
        is_unstructured: bool or list of bool, optional
            True if the corresponding array is unstructured. This keyword is
            ignored

        Returns
        -------
        %(Plotter.check_data.returns)s
        """
        if isinstance(name, six.string_types):
            name = [name]
            dims = [dims]
        N = len(name)
        if len(dims) != N:
            return [False] * N, [
                'Number of provided names (%i) and dimensions '
                '%(i) are not the same' % (N, len(dims))] * N
        checks = [True] * N
        messages = [''] * N
        for i, (n, d) in enumerate(zip(name, dims)):
            if len(n) == 0:
                checks[i] = False
                messages[i] = 'At least one variable name is required!'
            elif (not isstring(n) and len(n) != 1) and len(d) != 0:
                checks[i] = False
                messages[i] = 'Only one name is allowed per array!'
            elif len(d) == 0 or len(d) > 1:
                checks[i] = False
                messages[i] = 'Only 1-dimensional arrays are allowed!'
        return checks, messages


class ViolinPlotter(LinePlotter):
    """Plotter for making violin plots"""

    _rcparams_string = ['plotter.violin.']

    plot = ViolinPlot('plot')
    xlim = ViolinXlim('xlim')
    ylim = ViolinYlim('ylim')
    xticks = ViolinXTicks('xticks')
    xticklabels = ViolinXTickLabels('xticklabels')
    yticks = ViolinYTicks('yticks')
    yticklabels = ViolinYTickLabels('yticklabels')


class BarPlotter(ViolinPlotter):
    """Plotter for making bar plots"""

    _rcparams_string = ['plotter.bar.']

    plot = BarPlot('plot')
    xlim = BarXlim('xlim')
    ylim = BarYlim('ylim')
    xticklabels = BarXTickLabels('xticklabels')
    yticklabels = BarYTickLabels('yticklabels')


class Simple2DBase(Base2D):
    """Base class for :class:`Simple2DPlotter` and
    :class:`psyplot.plotter.maps.FieldPlotter` that defines the data
    management"""

    miss_color = MissColor('miss_color', index_in_list=0)

    @classmethod
    @docstrings.dedent
    def check_data(cls, name, dims, is_unstructured):
        """
        A validation method for the data shape

        Parameters
        ----------
        name: str or list of str
            The variable names (one variable per array)
        dims: list with length 1 or list of lists with length 1
            The dimension of the arrays. Only 1D-Arrays are allowed
        is_unstructured: bool or list of bool
            True if the corresponding array is unstructured.

        Returns
        -------
        %(Plotter.check_data.returns)s
        """
        if isinstance(name, six.string_types):
            name = [name]
            dims = [dims]
            is_unstructured = [is_unstructured]
        N = len(name)
        if N != 1:
            return [False] * N, [
                'Number of provided names (%i) must equal 1!' % (N)] * N
        elif len(dims) != 1:
            return [False], [
                'Number of provided dimension lists (%i) must equal 1!' % (
                    len(dims))]
        elif len(is_unstructured) != 1:
            return [False], [
                ('Number of provided unstructured information (%i) must '
                 'equal 1!') % (len(is_unstructured))]
        if len(name[0]) == 0:
            return [False], ['At least one variable name must be provided!']
        # unstructured arrays have only 1 dimension
        dimlen = 1 if is_unstructured[0] else 2
        # Check that the array is two-dimensional
        #
        # if more than one array name is provided, the dimensions should be
        # one les than dimlen to have a 2D array
        if (not isstring(name[0]) and len(name[0]) != 1 and
                len(dims[0]) != dimlen - 1):
            return [False], ['Only one name is allowed per array!']
        # otherwise the number of dimensions must equal dimlen
        if len(dims[0]) != dimlen:
            return [False], [
                'An array with dimension %i is required, not %i' % (
                    dimlen, len(dims[0]))]
        return [True], ['']

    def _set_data(self, *args, **kwargs):
        Plotter._set_data(self, *args, **kwargs)
        if isinstance(self.data, InteractiveList):
            data = self.data[0]
        else:
            data = self.data
        if not ((data.decoder.is_unstructured(data) and data.ndim == 1) or
                data.ndim == 2):
            raise ValueError("Can only plot 2-dimensional data!")


class Simple2DPlotter(Simple2DBase, LinePlotter):
    """Plotter for visualizing 2-dimensional data.

    See Also
    --------
    psyplot.plotter.maps.FieldPlotter"""

    transpose = Transpose('transpose')
    plot = SimplePlot2D('plot')
    xticks = XTicks2D('xticks')
    yticks = YTicks2D('yticks')
    xlim = Xlim2D('xlim')
    ylim = Ylim2D('ylim')
    legend = None
    legendlabels = None
    color = None  # no need for this formatoption


class BaseVectorPlotter(Base2D):
    """Base plotter for vector plots
    """

    _rcparams_string = ["plotter.vector."]

    arrowsize = ArrowSize('arrowsize')
    arrowstyle = ArrowStyle('arrowstyle')
    density = Density('density')
    color = VectorColor('color')
    linewidth = VectorLineWidth('linewidth')
    cbar = VectorCbar('cbar')
    bounds = VectorBounds('bounds')
    cticks = VectorCTicks('cticks')

    @classmethod
    @docstrings.dedent
    def check_data(cls, name, dims, is_unstructured):
        """
        A validation method for the data shape

        Parameters
        ----------
        name: str or list of str
            The variable names (two variables for the array or one if the dims
            are one greater)
        dims: list with length 1 or list of lists with length 1
            The dimension of the arrays. Only 2D-Arrays are allowed (or 1-D if
            the array is unstructured)
        is_unstructured: bool or list of bool
            True if the corresponding array is unstructured.

        Returns
        -------
        %(Plotter.check_data.returns)s
        """
        if isinstance(name, six.string_types):
            name = [name]
            dims = [dims]
            is_unstructured = [is_unstructured]
        N = len(name)
        if N != 1:
            return [False] * N, [
                'Number of provided names (%i) must equal 1!' % (N)] * N
        elif len(dims) != 1:
            return [False], [
                'Number of provided dimension lists (%i) must equal 1!' % (
                    len(dims))]
        elif len(is_unstructured) != 1:
            return [False], [
                ('Number of provided unstructured information (%i) must '
                 'equal 1!') % (len(is_unstructured))]
        if len(name[0]) == 0:
            return [False], ['Two variable names must be provided!']
        # unstructured arrays have only 1 dimension
        dimlen = 1 if is_unstructured[0] else 2
        # Check that the array is two-dimensional
        #
        # if more than one array name is provided, the dimensions should be
        # one les than dimlen to have a 2D array
        if (((isstring(name[0]) or len(name[0]) == 1) and
             len(dims[0]) != dimlen + 1) or len(name[0]) > 2):
            return [False], [
                ('Two variables (one for x- and one for y-direction) are '
                 'required!')]
        elif ((isstring(name[0]) or len(name[0]) == 1) and
              len(dims[0]) == dimlen + 1):
            dimlen += 1
        # otherwise the number of dimensions must equal dimlen
        if len(dims[0]) != dimlen:
            return [False], [
                'An array with dimension %i is required, not %i' % (
                    dimlen, len(dims[0]))]
        return [True], ['']

    def _set_data(self, *args, **kwargs):
        Plotter._set_data(self, *args, **kwargs)
        if isinstance(self.data, InteractiveList):
            data = self.data[0]
        else:
            data = self.data
        if not ((data.decoder.is_unstructured(data) and data.ndim == 2) or
                data.ndim == 3):
            raise ValueError("Can only plot 3-dimensional data!")


class SimpleVectorPlotter(BaseVectorPlotter, LinePlotter):
    """Plotter for visualizing 2-dimensional vector data

    See Also
    --------
    psyplot.plotter.maps.VectorPlotter"""

    plot = VectorPlot('plot')
    xticks = XTicks2D('xticks')
    yticks = YTicks2D('yticks')
    legend = None
    legendlabels = None


class CombinedBase(Plotter):
    """Base plotter for combined 2-dimensional scalar and vector plot"""
    _rcparams_string = ["plotter.combinedsimple."]

    cbar = Cbar('cbar', other_cbars=['vcbar'])
    cticks = CTicks('cticks')
    bounds = Bounds('bounds', index_in_list=0)
    arrowsize = ArrowSize('arrowsize', plot='vplot', index_in_list=1)
    arrowstyle = ArrowStyle('arrowstyle', plot='vplot', index_in_list=1)
    color = VectorColor('color', plot='vplot', cmap='vcmap', bounds='vbounds',
                        index_in_list=1)
    linewidth = VectorLineWidth('linewidth', plot='vplot', index_in_list=1)
    vcbar = VectorCbar('vcbar', plot='vplot', cmap='vcmap', bounds='vbounds',
                       cbarspacing='vcbarspacing', other_cbars=['cbar'],
                       index_in_list=1)
    vcbarspacing = CbarSpacing('vcbarspacing', cbar='vcbar', index_in_list=1)
    vclabel = VCLabel('vclabel', plot='vplot', cbar='vcbar', index_in_list=1)
    vclabelsize = label_size(vclabel, 'Vector colorbar label',
                             dependencies=['vclabel'])
    vclabelweight = label_weight(vclabel, 'Vector colorbar label',
                                 dependencies=['vclabel'])
    vclabelprops = label_props(vclabel, 'Vector colorbar label',
                               dependencies=['vclabel'])
    vcmap = CMap('vcmap', index_in_list=1)
    vbounds = VectorBounds('vbounds', index_in_list=1)
    vcticks = VectorCTicks('vcticks', cbar='vcbar', plot='vplot',
                           bounds='vbounds', index_in_list=1)
    vcticklabels = CTickLabels('vcticklabels', cbar='vcbar', index_in_list=1)
    vcticksize = CTickSize('vcticksize', cbar='vcbar', index_in_list=1)
    vctickweight = CTickWeight('vctickweight', cbar='vcbar', index_in_list=1)
    vctickprops = CTickProps('vctickprops', cbar='vcbar',
                             ticksize='vcticksize', index_in_list=1)
    # make sure that masking options only affect the scalar field
    maskless = MaskLess('maskless', index_in_list=0)
    maskleq = MaskLeq('maskleq', index_in_list=0)
    maskgreater = MaskGreater('maskgreater', index_in_list=0)
    maskgeq = MaskGeq('maskgeq', index_in_list=0)
    maskbetween = MaskBetween('maskbetween', index_in_list=0)

    @classmethod
    @docstrings.dedent
    def check_data(cls, name, dims, is_unstructured):
        """
        A validation method for the data shape

        Parameters
        ----------
        name: list of str with length 2
            The variable names (one for the first, two for the second array)
        dims: list with length 2 of lists with length 1
            The dimension of the arrays. Only 2D-Arrays are allowed (or 1-D if
            an array is unstructured)
        is_unstructured: bool or list of bool
            True if the corresponding array is unstructured.

        Returns
        -------
        %(Plotter.check_data.returns)s
        """
        if isinstance(name, six.string_types):
            name = [name]
            dims = [dims]
            is_unstructured = [is_unstructured]
        msg = ('Two arrays are required (one for the scalar and '
               'one for the vector field)')
        if len(name) < 2:
            return [None], [msg]
        elif len(name) > 2:
            return [False], [msg]
        valid1, msg1 = Simple2DBase.check_data(name[:1], dims[0:1],
                                               is_unstructured[:1])
        valid2, msg2 = BaseVectorPlotter.check_data(name[1:], dims[1:],
                                                    is_unstructured[1:])
        return valid1 + valid2, msg1 + msg2

    def _set_data(self, *args, **kwargs):
        super(CombinedBase, self)._set_data(*args, **kwargs)
        # implement 2 simple checks to make sure that we get the right data
        if not isinstance(self.plot_data, InteractiveList):
            raise ValueError(
                "Combined plots must be lists of one scalar field and a"
                "vector field. Got one %s instead" % str(type(
                    self.plot_data)))
        elif len(self.plot_data) < 2:
            raise ValueError(
                "Combined plots must be lists of one scalar field and a"
                "vector field. Got a list of length %i instead!" % len(
                    self.plot_data))


class CombinedSimplePlotter(CombinedBase, Simple2DPlotter,
                            SimpleVectorPlotter):
    """Combined 2D plotter and vector plotter

    See Also
    --------
    psyplot.plotter.maps.CombinedPlotter: for visualizing the data on a map"""
    plot = Plot2D('plot', index_in_list=0)
    vplot = CombinedVectorPlot('vplot', index_in_list=1, cmap='vcmap',
                       bounds='vbounds')
    density = Density('density', plot='vplot', index_in_list=1)
