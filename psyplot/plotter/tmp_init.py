"""Core package for interactive visualization in the psyplot package

This package defines the :class:`Plotter` and :class:`Formatoption` classes,
the core of the visualization in the :mod:`psyplot` package. Each
:class:`Plotter` combines a set of formatoption keys where each formatoption
key is represented by a :class:`Formatoption` subclass.

Important plotters are

.. autosummary::

    simpleplotter.SimplePlotter
    simpleplotter.Simple2DPlotter
    simpleplotter.SimpleVectorPlotter
    maps.FieldPlotter
    maps.VectorPlotter
    maps.CombinedPlotter"""

import six
from abc import ABCMeta, abstractmethod
from textwrap import TextWrapper
from itertools import chain, groupby, tee, repeat, starmap
from collections import defaultdict
from difflib import get_close_matches
from threading import RLock
from datetime import datetime, timedelta
from numpy import datetime64, timedelta64, ndarray
from xray.core.formatting import format_timestamp, format_timedelta
from .. import rcParams
from ..warning import warn, critical, PsyPlotRuntimeWarning
from ..compat.pycompat import map, filter, filterfalse, zip, range
from ..config.rcsetup import defaultParams, SubDict
from ..docstring import docstrings, dedent
from ..data import InteractiveList


#: if True, include the link in the formatoption table of the
#: :attr:`Plotter.keys` method (i.e. print ":attr:`~psyplot.plotter.maps.title`"
#: instead of "title" in the output string). This is set to True when the doc
#: is built to make the formatoptions in the plotting methods of the
#: :class:`syplot.project.ProjectPlotter` class link to the right formatoption
#: Otherwise it is set to False because it would be an overkill
_fmt_links = False


#: :class:`dict`. Mapping from group to group names
groups = {
    'axes': 'Axes formatoptions',
    'labels': 'Label formatoptions',
    'plot': 'Plot formatoptions',
    'colors': 'Color coding formatoptions',
    'misc': 'Miscallaneous formatoptions',
    'ticks': 'Axis tick formatoptions',
    'vector': 'Vector plot formatoptions',
    'masking': 'Masking formatoptions'
    }


def _identity(obj):
    """identity function to make no validation"""
    return obj


def format_time(x):
    """Formats date values

    This function formats :class:`datetime.datetime` and
    :class:`datetime.timedelta` objects (and the corresponding numpy objects)
    using the :func:`xray.core.formatting.format_timestamp` and the
    :func:`xray.core.formatting.format_timedelta` functions.

    Parameters
    ----------
    x: object
        The value to format. If not a time object, the value is returned

    Returns
    -------
    str or `x`
        Either the formatted time object or the initial `x`"""
    if isinstance(x, (datetime64, datetime)):
        return format_timestamp(x)
    if isinstance(x, (timedelta64, timedelta)):
        return format_timedelta(x)
    if isinstance(x, ndarray):
        return list(x) if x.ndim else x[()]
    return x


def is_data_dependent(fmto, data):
    """Check whether a formatoption is data dependent

    Parameters
    ----------
    fmto: Formatoption
        The :class:`Formatoption` instance to check
    data: xray.DataArray
        The data array to use if the :attr:`~Formatoption.data_dependent`
        attribute is a callable

    Returns
    -------
    bool
        True, if the formatoption depends on the data"""
    if callable(fmto.data_dependent):
        return fmto.data_dependent(data)
    return fmto.data_dependent


@docstrings.get_sectionsf('check_key', sections=['Parameters', 'Returns',
                                                 'Raises'])
@dedent
def check_key(key, possible_keys, raise_error=True,
              name='formatoption keyword', *args, **kwargs):
    """
    Checks whether the key is in a list of possible keys

    This function checks whether the given `key` is in `possible_keys` and if
    not looks for similar sounding keys

    Parameters
    ----------
    key: str
        Key to check
    possible_keys: list of strings
        a list of possible keys to use
    raise_error: bool
        If not True, a list of similar keys is returned
    name: str
        The name of the key that shall be used in the error message
    ``*args`` and ``**kwargs``
        They are passed to the :func:`difflib.get_close_matches` function
        (i.e. `n` to increase the number of returned similar keys and
        `cutoff` to change the sensibility)

    Returns
    -------
    bool
        True if the `key` is a valid string, else False

    list
        A list of similar formatoption strings (if found)
    str
        An error message which includes

    Raises
    ------
    KeyError
        If the key is not a valid formatoption and `raise_error` is True"""
    if key not in possible_keys:
        similarkeys = get_close_matches(key, possible_keys, *args, **kwargs)
        if similarkeys:
            msg = ('Unknown %s %s! Possible similiar '
                   'frasings are %s.') % (name, key, ', '.join(similarkeys))
        else:
            msg = ("Unknown %s %s! See show_fmtkeys "
                   "function for possible formatopion keywords") % (name, key)
        if not raise_error:
            return False, similarkeys, msg
        raise KeyError(msg)
    else:
        return True, [key], ''


class _TempBool(object):
    """Wrapper around a boolean defining an __enter__ and __exit__ method

    Parameters
    ----------
    value: bool
        value of the object"""

    #: default boolean value for the :attr:`value` attribute
    default = False

    #: boolean value indicating whether there shall be a validation or not
    value = False

    def __init__(self, default=False):
        """
        Parameters
        ----------
        default: bool
            value of the object"""
        self.default = default
        self.value = default

    def __enter__(self):
        self.value = not self.default

    def __exit__(self, type, value, tb):
        self.value = self.default

    if six.PY2:
        def __nonzero__(self):
            return self.value
    else:
        def __bool__(self):
            return self.value


def _child_property(childname):
    def get_x(self):
        return getattr(self.plotter, self._child_mapping[childname])

    return property(
        get_x, doc=childname + " Formatoption instance in the plotter")


class FormatoptionMeta(ABCMeta):
    """Meta class for formatoptions

    This class serves as a meta class for formatoptions and allows a more
    efficient docstring generation by using the
    :attr:`psyplot.docstring.docstrings` when creating a new formatoption
    class"""
    def __new__(cls, clsname, bases, dct):
        """Assign an automatic documentation to the formatoption"""
        dct['__doc__'] = docstrings.dedents(dct.get('__doc__'))
        new_cls = super(FormatoptionMeta, cls).__new__(cls, clsname, bases,
                                                       dct)
        for childname in chain(new_cls.children, new_cls.dependencies,
                               new_cls.connections, new_cls.parents):
            setattr(new_cls, childname, _child_property(childname))
        if new_cls.plot_fmt:
            new_cls.data_dependent = True
        return new_cls


# priority values

#: Priority value of formatoptions that are updated before the data is loaded.
START = 30
#: Priority value of formatoptions that are updated before the plot it made.
BEFOREPLOTTING = 20
#: Priority value of formatoptions that are updated at the end.
END = 10


@six.add_metaclass(FormatoptionMeta)
class Formatoption(object):
    """Abstract formatoption

    This class serves as an abstract version of an formatoption descriptor
    that can be used by :class:`~psyplot.plotter.Plotter` instances."""

    priority = END
    """:class:`int`. Priority value of the the formatoption determining when
    the formatoption is updated.

    - 10: at the end (for labels, etc.)
    - 20: before the plotting (e.g. for colormaps, etc.)
    - 30: before loading the data (e.g. for lonlatbox)"""

    #: :class:`str`. Formatoption key of this class in the
    #: :class:`~psyplot.plotter.Plotter` class
    key = None

    #: :class:`~psyplot.plotter.Plotter`. Plotter instance this formatoption
    #: belongs to
    plotter = None

    #: `list of str`. List of formatoptions that have to be updated before this
    #: one is updated. Those formatoptions are only updated if they exist in
    #: the update parameters.
    children = []

    #: `list of str`. List of formatoptions that force an update of this
    #: formatoption if they are updated.
    dependencies = []

    #: `list of str`. Connections to other formatoptions that are (different
    #: from :attr:`dependencies` and :attr:`children`) not important for the
    #: update process
    connections = []

    #: `list of str`. List of formatoptions that, if included in the update,
    #: prevent the update of this formatoption.
    parents = []

    #: :class:`bool`. Has to be True if the formatoption has a ``make_plot``
    #: method to make the plot.
    plot_fmt = False

    #: :class:`bool`. True if an update of this formatoption requires a
    #: clearing of the axes and reinitializing of the plot
    requires_clearing = False

    #: :class:`str`. Key of the group name in :data:`groups` of this
    #: formatoption keyword
    group = 'misc'

    #: :class:`bool` or a callable. This attribute indicates whether this
    #: :class:`Formatoption` depends on the data and should be updated if the
    #: data changes. If it is a callable, it must accept one argument: the
    #: new data. (Note: This is automatically set to True for plot
    #: formatoptions)
    data_dependent = False

    #: :class:`bool`. True if this formatoption needs an update after the plot
    #: has changed
    update_after_plot = False

    #: :class:`set` of the :class:`Formatoption` instance that are shared
    #: with this instance.
    shared = set()

    #: int or None. Index that is used in case the plotting data is a
    #: :class:`psyplot.InteractiveList`
    index_in_list = 0

    @property
    def init_kwargs(self):
        """:class:`dict` key word arguments that are passed to the
        initialization of a new instance when accessed from the descriptor"""
        return self._child_mapping

    @property
    def project(self):
        """Project of the plotter of this instance"""
        return self.plotter.project

    @property
    def ax(self):
        """The axes this Formatoption plots on"""
        return self.plotter.ax

    @property
    def lock(self):
        """A :class:`threading.Rlock` instance to lock while updating

        This lock is used when multiple :class:`plotter` instances are
        updated at the same time while sharing formatoptions."""
        try:
            return self._lock
        except AttributeError:
            self._lock = RLock()
            return self._lock

    @property
    def logger(self):
        """Logger of the plotter"""
        return self.plotter.logger.getChild(self.key)

    @property
    def groupname(self):
        """Long name of the group this formatoption belongs too."""
        try:
            return groups[self.group]
        except KeyError:
            warn("Unknown formatoption group " + str(self.group),
                 PsyPlotRuntimeWarning)
            return "Unknown"

    @property
    def raw_data(self):
        """The full :class:`psyplot.InteractiveArray` of this plotter"""
        if self.index_in_list is not None and isinstance(
                self.plotter.data, InteractiveList):
            return self.plotter.data[self.index_in_list]
        else:
            return self.plotter.data

    @property
    def data(self):
        """The :class:`psyplot.DataArray` that is plotted"""
        if self.index_in_list is not None and isinstance(
                self.plotter.data, InteractiveList):
            return self.plotter.plot_data[self.index_in_list]
        else:
            return self.plotter.plot_data

    @data.setter
    def data(self, value):
        if self.index_in_list is not None and isinstance(
                self.plotter.plot_data, InteractiveList):
            self.plotter.plot_data[self.index_in_list] = value
        else:
            self.plotter.plot_data = value

    @property
    def iter_data(self):
        """Returns an iterator over the data arrays"""
        if isinstance(self.data, InteractiveList):
            return iter(self.data)
        return iter([self.data])

    @property
    def iter_plotdata(self):
        """Returns an iterator over the plot data arrays"""
        if isinstance(self.plotter.plot_data, InteractiveList):
            return iter(self.plotter.plot_data)
        return iter([self.plotter.plot_data])

    @property
    def validate(self):
        """Static validation method of the formatoption"""
        try:
            return self._validate
        except AttributeError:
            try:
                self._validate = self.plotter.get_vfunc(self.key)
            except KeyError:
                warn("Could not find a validation function for %s "
                     "formatoption keyword! No validation will be made!" % (
                         self.key))
                self._validate = _identity
        return self._validate

    @validate.setter
    def validate(self, value):
        self._validate = value

    @property
    def default(self):
        """Default value of this formatoption"""
        return self.plotter.rc[self.key]

    @property
    def default_key(self):
        """The key of this formatoption in the :attr:`psyplot.rcParams`"""
        return self.plotter.rc._get_val_and_base(self.key)[0]

    @property
    def value(self):
        """Value of the formatoption in the corresponding :attr:`plotter` or
        the shared value"""
        if self.key in self.plotter._shared:
            return self.plotter._shared[self.key].value2share
        return self.plotter[self.key]

    @property
    @dedent
    def changed(self):
        """
        :class:`bool` indicating whether the value changed compared to the
        default or not."""
        return self.diff(self.default)

    @property
    @dedent
    def value2share(self):
        """
        The value that is passed to shared formatoptions (by default, the
        :attr:`value` attribute)"""
        return self.value

    @docstrings.get_sectionsf('Formatoption')
    @dedent
    def __init__(self, key, plotter=None, index_in_list=None, **kwargs):
        """
        Parameters
        ----------
        key: str
            formatoption key in the `plotter`
        plotter: psyplot.plotter.Plotter
            Plotter instance that holds this formatoption. If None, it is
            assumed that this instance serves as a descriptor.
        index_in_list: int or None
            The index that shall be used if the data is a
            :class:`psyplot.InteractiveList`
        ``**kwargs``
            Further keywords may be used to specify different names for
            children, dependencies and connection formatoptions that match the
            setup of the plotter. Hence, keywords may be anything of the
            :attr:`children`, :attr:`dependencies` and :attr:`connections`
            attributes, with values being the name of the new formatoption in
            this plotter."""
        self.key = key
        self.plotter = plotter
        self.index_in_list = index_in_list
        self.shared = set()
        self._child_mapping = dict(zip(*tee(chain(
            self.children, self.dependencies, self.connections,
            self.parents), 2)))
        # check kwargs
        for key in (key for key in kwargs if key not in self._child_mapping):
            raise TypeError(
                '%s.__init__() got an unexpected keyword argument %r' % (
                    self.__class__.__name__, key))
        # set up child mapping
        self._child_mapping.update(kwargs)
        # reset the dependency lists to match the current plotter setup
        for attr in ['children', 'dependencies', 'connections', 'parents']:
            setattr(self, attr, list(map(lambda key: self._child_mapping[key],
                                         getattr(self, attr))))

    def __set__(self, instance, value):
        if isinstance(value, Formatoption):
            setattr(instance, '_' + self.key, value)
            return
        fmto = getattr(instance, self.key)
        fmto.set_value(value)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        try:
            return getattr(instance, '_' + self.key)
        except AttributeError:
            fmto = self.__class__(
                self.key, instance, self.index_in_list, **self.init_kwargs)
            setattr(instance, '_' + self.key, fmto)
            return fmto

    def __delete__(self, instance, owner):
        fmto = getattr(instance, '_' + self.key)
        with instance.no_validation:
            instance[self.key] = fmto.default

    @docstrings.get_sectionsf('Formatoption.set_value')
    @dedent
    def set_value(self, value, validate=True, todefault=False):
        """
        Set (and validate) the value in the plotter

        Parameters
        ----------
        value
            Value to set
        validate: bool
            if True, validate the `value` before it is set
        todefault: bool
            True if the value is updated to the default value"""
        # do nothing if the key is shared
        if self.key in self.plotter._shared:
            return
        with self.plotter.no_validation:
            try:
                self.plotter[self.key] = value if not validate else \
                    self.validate(value)
            except ValueError:
                self.logger.error("Error while setting %s!" % self.key,
                                          exc_info=True)

    def check_and_set(self, value, todefault=False):
        """Checks the value and sets the value if it changed

        This method checks the value and sets it only if the :meth:`diff`
        method result of the given `value` is True

        Parameters
        ----------
        value
            A possible value to set
        todefault: bool
            True if the value is updated to the default value

        Returns
        -------
        bool
            A boolean to indicate whether it has been set or not"""
        value = self.validate(value)
        if self.diff(value):
            self.set_value(value, validate=False, todefault=todefault)
            return True
        return False

    def diff(self, value):
        """Checks whether the given value differs from what is currently set

        Parameters
        ----------
        value
            A possible value to set (make sure that it has been validate via
            the :attr:`validate` attribute before)

        Returns
        -------
        bool
            True if the value differs from what is currently set"""
        return value != self.value

    def initialize_plot(self, value):
        """Method that is called when the plot is made the first time

        Parameters
        ----------
        value
            The value to use for the initialization"""
        self.update(value)

    @abstractmethod
    def update(self, value):
        """Method that is call to update the formatoption on the axes

        Parameters
        ----------
        value
            Value to update"""
        pass

    def share(self, fmto, initializing=False, **kwargs):
        """Share the settings of this formatoption with other data objects

        Parameters
        ----------
        fmto: Formatoption
            The :class:`Formatoption` instance to share the attributes with
        ``**kwargs``
            Any other keyword argument that shall be passed to the update
            method of `fmto`"""
        # lock all  the childrens and the formatoption itself
        fmto._lock_children()
        fmto.lock.acquire()
        # update the other plotter
        if initializing:
            fmto.initialize_plot(self.value2share, **kwargs)
        else:
            fmto.update(self.value2share, **kwargs)
        self.shared.add(fmto)
        # release the locks
        fmto.lock.release()
        fmto._release_children()

    def _lock_children(self):
        """acquire the locks of the children"""
        plotter = self.plotter
        for key in self.children:
            try:
                getattr(plotter, key).lock.acquire()
            except AttributeError:
                pass

    def _release_children(self):
        """release the locks of the children"""
        plotter = self.plotter
        for key in self.children:
            try:
                getattr(plotter, key).lock.release()
            except AttributeError:
                pass

    def finish_update(self):
        """Finish the update and sharing process"""
        pass

    @dedent
    def remove(self):
        """
        Method to remove the effects of this formatoption

        This method is called when the axes is cleared due to a
        formatoption with :attr:`requires_clearing` set to True. You don't
        necessarily have to implement this formatoption if your plot results
        are removed by the usual :meth:`matplotlib.axes.Axes.clear` method."""
        pass


class DictFormatoption(Formatoption):
    """
    Base formatoption class defining an alternative set_value that works for
    dictionaries."""

    @docstrings.dedent
    def set_value(self, value, validate=True, todefault=False):
        """
        Set (and validate) the value in the plotter

        Parameters
        ----------
        %(Formatoption.set_value.parameters)s

        Notes
        -----
        - If the current value in the plotter is None, then it will be set with
          the given `value`, otherwise the current value in the plotter is
          updated
        - If the value is an empty dictionary, the value in the plotter is
          cleared"""
        value = value if not validate else self.validate(value)
        # if the key in the plotter is not already set (i.e. it is initialized
        # with None, we set it)
        if self.plotter[self.key] is None:
            with self.plotter.no_validation:
                self.plotter[self.key] = value.copy()
        # in case of an empty dict, clear the value
        elif not value:
            self.plotter[self.key].clear()
        # otherwhise we update the dictionary
        else:
            if todefault:
                self.plotter[self.key].clear()
            self.plotter[self.key].update(value)


@docstrings.get_sectionsf('Plotter')
class Plotter(dict):
    """Interactive plotting object for one or more data arrays

    This class is the base for the interactive plotting with the psyplot module.
    It capabilities are determined by it's descriptor classes that are
    derived from the :class:`Formatoption` class"""

    #: List of base strings in the :attr:`psyplot.rcParams` dictionary
    _rcparams_string = []

    @property
    def no_validation(self):
        """Temporarily disable the validation

        Examples
        --------
        Although it is not recommended to set a value with disabled validation,
        you can disable it via::

            >>> with plotter.no_validation:
            ...     plotter['ticksize'] = 'x'

        To permanently disable the validation, simply set

            >>> plotter.no_validation = True
            >>> plotter['ticksize'] = 'x'
            >>> plotter.no_validation = False  # reenable validation"""
        try:
            return self._no_validation
        except AttributeError:
            self._no_validation = _TempBool()
            return self._no_validation

    @no_validation.setter
    def no_validation(self, value):
        self.no_validation.value = bool(value)

    @property
    def ax(self):
        """Axes instance of the plot"""
        if self._ax is None:
            import matplotlib.pyplot as plt
            plt.figure()
            self._ax = plt.axes()
        return self._ax

    @ax.setter
    def ax(self, value):
        self._ax = value

    #: The :class:`psyplot.project.Project` instance this plotter belongs to
    project = None

    @property
    @dedent
    def rc(self):
        """
        Default values for this plotter

        This :class:`~psyplot.config.rcsetup.SubDict` stores the default values
        for this plotter. A modification of the dictionary does not affect
        other plotter instances unless you set the
        :attr:`~psyplot.config.rcsetup.SubDict.trace` attribute to True"""
        try:
            return self._rc
        except AttributeError:
            self._set_rc()
            return self._rc

    @property
    def base_variables(self):
        """A mapping from the base_variable names to the variables"""
        if isinstance(self.data, InteractiveList):
            return dict(chain(*map(
                lambda arr: six.iteritems(arr.base_variables),
                self.data)))
        else:
            return self.data.base_variables

    @property
    def auto_update(self):
        """:class:`bool`. Boolean controlling whether the :meth:`start_update`
        method is automatically called by the :meth:`update` method"""
        return self._auto_update

    @auto_update.setter
    def auto_update(self, value):
        self._auto_update = value

    @property
    def changed(self):
        """:class:`dict` containing the key value pairs that are not the
        default"""
        return {key: value for key, value in six.iteritems(self)
                if getattr(self, key).changed}

    @property
    def figs2draw(self):
        """All figures that have been manipulated through sharing and the own
        figure.

        Notes
        -----
        Using this property set will reset the figures too draw"""
        return self._figs2draw.union([self.ax.get_figure()])

    @property
    @docstrings
    def njobs(self):
        """%(InteractiveBase.njobs)s"""
        if self.disabled:
            return 0
        return 1

    @property
    def _fmtos(self):
        """Iterator over the formatoptions"""
        return (getattr(self, key) for key in self)

    @property
    def _fmto_groups(self):
        """Mapping from group to a set of formatoptions"""
        ret = defaultdict(set)
        for key in self:
            ret[getattr(self, key).group].add(getattr(self, key))
        return dict(ret)

    @property
    def fmt_groups(self):
        """A mapping from the formatoption group to the formatoptions"""
        ret = defaultdict(set)
        for key in self:
            ret[getattr(self, key).group].add(key)
        return dict(ret)

    @property
    def data(self):
        """The :class:`psyplot.InteractiveBase` instance of this plotter"""
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self.set_logger(force=True)

    @property
    def plot_data(self):
        """The data that is used for plotting"""
        return getattr(self, '_plot_data', self.data)

    @plot_data.setter
    def plot_data(self, value):
        if isinstance(value, InteractiveList):
            self._plot_data = value.copy()
        else:
            self._plot_data = value

    docstrings.keep_params('InteractiveBase.parameters', 'auto_update')

    @docstrings.get_sectionsf('Plotter')
    @docstrings.dedent
    def __init__(self, data=None, ax=None, auto_update=None, project=None,
                 draw=True, make_plot=True, clear=False, **kwargs):
        """
        Parameters
        ----------
        data: InteractiveArray or ArrayList, optional
            Data object that shall be visualized. If given and `plot` is True,
            the :meth:`initialize_plot` method is called at the end. Otherwise
            you can call this method later by yourself
        ax: matplotlib.axes.Axes
            Matplotlib Axes to plot on. If None, a new one will be created as
            soon as the :meth:`initialize_plot` method is called
        %(InteractiveBase.parameters.auto_update)s
        %(InteractiveBase.start_update.parameters.draw)s
        make_plot: bool
            If True, and `data` is not None, the plot is initialized. Otherwise
            only the framework between plotter and data is set up
        clear: bool
            If True, the axes is cleared first
        ``**kwargs``
            Any formatoption key from the :attr:`formatoptions` attribute that
            shall be used"""
        self.project = project
        self.ax = ax
        self.data = data
        if auto_update is None:
            auto_update = rcParams['lists.auto_update']
        self.auto_update = auto_update
        self._registered_updates = {}
        self._todefault = False
        self._old_fmt = []
        self._figs2draw = set()
        self.disabled = False
        #: Dictionary holding the Formatoption instances of other plotters
        #: if their value shall be used instead of the one in this instance
        self._shared = {}
        #: list of str. Formatoption keys that were changed during the last
        #: update
        self._last_update = []
        self.replot = True
        self.cleared = clear

        with self.no_validation:
            # first set the formatoptions from the class descriptors by using
            # the this class and the base classes
            for key in self._get_formatoptions():
                self[key] = None
        for key, value in six.iteritems(self.rc):
            self._try2set(getattr(self, key), value, validate=False)
        for key, value in six.iteritems(kwargs):
            self[key] = value
        self.initialize_plot(data, ax=ax, draw=draw, clear=clear,
                             make_plot=make_plot)

    def _try2set(self, fmto, *args, **kwargs):
        """Sets the value in `fmto` and gives additional informations when fail

        Parameters
        ----------
        fmto: Formatoption
        ``*args`` and ``**kwargs``
            Anything that is passed to `fmto`s :meth:`~Formatoption.set_value`
            method"""
        try:
            fmto.set_value(*args, **kwargs)
        except:
            critical("Error while setting %s!" % fmto.key,
                     logger=getattr(self, 'logger', None))
            raise

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            self.check_key(key)

    def __setitem__(self, key, value):
        if not self.no_validation:
            self.check_key(key)
            self._try2set(getattr(self, key), value)
            return
        # prevent from setting during an update process
        getattr(self, key).lock.acquire()
        dict.__setitem__(self, key, value)
        getattr(self, key).lock.release()

    def __delitem__(self, key):
        self[key] = getattr(self, key).default

    docstrings.delete_params('check_key.parameters', 'possible_keys', 'name')

    @docstrings.dedent
    def check_key(self, key, raise_error=True, *args, **kwargs):
        """
        Checks whether the key is a valid formatoption

        Parameters
        ----------
        %(check_key.parameters.no_possible_keys|name)s

        Returns
        -------
        %(check_key.returns)s

        Raises
        ------
        %(check_key.raises)s"""
        return check_key(
            key, possible_keys=list(self), raise_error=raise_error,
            name='formatoption keyword', *args, **kwargs)


    docstrings.keep_params('Plotter.parameters', 'ax', 'make_plot', 'clear')

    @docstrings.dedent
    def initialize_plot(self, data=None, ax=None, make_plot=True, clear=False,
                        draw=True, remove=False):
        """
        Initialize the plot for a data array

        Parameters
        ----------
        data: InteractiveArray or ArrayList, optional
            Data object that shall be visualized.

            - If not None and `plot` is True, the given data is visualized.
            - If None and the :attr:`data` attribute is not None, the data in
              the :attr:`data` attribute is visualized
            - If both are None, nothing is done.
        %(Plotter.parameters.ax|make_plot|clear)s
        %(InteractiveBase.start_update.parameters.draw)s
        remove: bool
            If True, old effects by the formatoptions in this plotter are
            undone first"""
        if data is None and self.data is not None:
            data = self.data
        else:
            self.data = data
        self.ax = ax
        if data is None:  # nothing to do if no data is given
            return
        self.auto_update = self.auto_update or data.auto_update
        data.plotter = self
        if not make_plot:  # stop here if we shall not plot
            return
        self.logger.debug("Initializing plot...")
        if remove:
            self.logger.debug("    Removing old formatoptions...")
            for fmto in self._fmtos:
                fmto.remove()
        if clear:
            self.logger.debug("    Clearing axes...")
            self.ax.clear()
        fmto_groups = self._grouped_fmtos(self._sorted_by_priority(
            list(self._fmtos)))
        self.plot_data = self.data
        for priority, grouper in fmto_groups:
            self._plot_by_priority(priority, grouper, initializing=True)
        self.cleared = False
        self.replot = False

        if draw:
            self.draw()

    docstrings.keep_params('InteractiveBase._register_update.parameters',
                           'todefault')

    @docstrings.get_sectionsf('Plotter._register_update')
    @docstrings.dedent
    def _register_update(self, fmt={}, replot=False, todefault=False):
        """
        Register formatoptions for the update

        Parameters
        ----------
        fmt: dict
            Keys can be any valid formatoptions with the corresponding values
            (see the :attr:`formatoptions` attribute)
        replot: bool
            Boolean that determines whether the data specific formatoptions
            shall be updated in any case or not.
        %(InteractiveBase._register_update.parameters.todefault)s"""
        if self.disabled:
            return
        self.replot = self.replot or replot
        self._todefault = self._todefault or todefault
        # check the keys
        list(map(self.check_key, fmt))
        self._registered_updates.update(fmt)

    @docstrings.dedent
    def start_update(self, draw=True, queue=None):
        """
        Conduct the registered plot updates

        This method starts the updates from what has been registered by the
        :meth:`update` method. You can call this method if you did not set the
        `auto_update` parameter when calling the :meth:`update` method and when
        the :attr:`auto_update` attribute is False.

        Parameters
        ----------
        %(InteractiveBase.start_update.parameters)s

        Returns
        -------
        %(InteractiveBase.start_update.returns)s

        See Also
        --------
        :attr:`auto_update`, update"""
        if self.disabled:
            return False

        if queue is not None:
            queue.get()
        # lock all formatoptions
        for fmto in self._fmtos:
            fmto.lock.acquire()
        if queue is not None:
            queue.task_done()
            # wait for the other tasks to finish
            queue.join()

        # update the formatoptions
        self._save_state()
        try:
            self.logger.debug("Starting update of %r",
                              self._registered_updates.keys())
            fmto_groups = self._grouped_fmtos(self._sorted_by_priority(
                self._set_and_filter()))
        except:
            # restore last (working) state
            last_state = self._old_fmt.pop(-1)
            with self.no_validation:
                for key in self:
                    self[key] = last_state.get(key, getattr(self, key).default)
            # raise the error
            raise
        else:
            # if any formatoption requires a clearing of the axes is updated,
            # we reinitialize the plot
            if self.cleared:
                self.reinit(draw=draw)
                return True
            # otherwise we update it
            arr_draw = False
            for priority, grouper in fmto_groups:
                arr_draw = True
                self._plot_by_priority(priority, grouper)
            if draw and arr_draw:
                self.draw()
            self.replot = False
            return arr_draw
        finally:
            # make sure that all locks are released
            for fmto in self._fmtos:
                fmto.finish_update()
                try:
                    fmto.lock.release()
                except RuntimeError:
                    pass

    def _plot_by_priority(self, priority, fmtos, initializing=False):
        def update_and_or_share(fmto):
            other_fmto = self._shared.get(fmto.key)
            # although shared fmtos have been filtered out in the
            # _set_and_filter method, they might have come in again because
            # they are data_dependent
            if other_fmto:
                self.logger.debug("%s is shared with %s", fmto.key,
                                  other_fmto.plotter.logger.name)
                other_fmto.share(fmto, initializing=initializing)
            # but if not, share them
            else:
                if initializing:
                    self.logger.debug("Initializing %s", fmto.key)
                    fmto.initialize_plot(fmto.value)
                else:
                    self.logger.debug("Updating %s", fmto.key)
                    fmto.update(fmto.value)
                for fmto2 in fmto.shared:
                    self.logger.debug("    Updating shared %s: %s",
                                      fmto.key, fmto2.plotter.logger.name)
                    fmto.share(fmto2, initializing=initializing)
                    self._figs2draw.update(fmto2.plotter.figs2draw)
                    fmto2.plotter._figs2draw.clear()
            try:
                fmto.lock.release()
            except RuntimeError:
                pass

        self._initializing = initializing

        self.logger.debug(
            "%s formatoptions with priority %i",
            "Initializing" if initializing else "Updating", priority)

        if priority >= START or priority == END:
            for fmto in fmtos:
                update_and_or_share(fmto)
        elif priority == BEFOREPLOTTING:
            for fmto in fmtos:
                update_and_or_share(fmto)
            self._make_plot()

        self._initializing = False

    @docstrings.dedent
    def reinit(self, draw=True):
        """
        Reinitializes the plot with the same data and on the same axes.

        Parameters
        ----------
        %(InteractiveBase.start_update.parameters.draw)s

        Warnings
        --------
        The axes is cleared when calling this method!"""
        # call the initialize_plot method. Note that clear can be set to
        # False if any fmto has requires_clearing attribute set to True,
        # because this then has been cleared before
        self.initialize_plot(
            self.data, self.ax, draw=draw, clear=not any(
                fmto.requires_clearing for fmto in self._fmtos),
            remove=True)

    def draw(self):
        """Draw the figures and those that are shared and have been changed"""
        for fig in self.figs2draw:
            fig.canvas.draw()
        self._figs2draw.clear()

    def _grouped_fmtos(self, fmtos):
        def key_func(fmto):
            if fmto.priority >= START:
                return START
            elif fmto.priority >= BEFOREPLOTTING:
                return BEFOREPLOTTING
            else:
                return END
        return groupby(fmtos, key_func)

    def _set_and_filter(self):
        """Filters the registered updates and sort out what is not needed

        This method filters out the formatoptions that have not changed, sets
        the new value and returns an iterable that is sorted by the priority
        (highest priority comes first) and dependencies

        Returns
        -------
        list
            list of :class:`Formatoption` objects that have to be updated"""
        fmtos = []
        seen = set()
        for key, value in chain(
                six.iteritems(self._registered_updates),
                six.iteritems(self.rc) if self._todefault else ()):
            if key in seen:
                continue
            seen.add(key)
            fmto = getattr(self, key)
            # if the key is shared, a warning will be printed later
            if key in self._shared:
                warn(("%s formatoption is shared with another plotter."
                      " Use the unshare method to enable the updating") % (
                          fmto.key),
                     logger=self.logger)
                changed = False
            else:
                try:
                    changed = fmto.check_and_set(
                        value, todefault=self._todefault)
                except:
                    self._registered_updates.pop(key, None)
                    raise
            if changed:
                fmtos.append(fmto)
        fmtos = self._insert_additionals(fmtos, seen)
        self._todefault = False
        self._registered_updates.clear()
        return fmtos

    def _insert_additionals(self, fmtos, seen=None):
        """
        Insert additional formatoptions into `fmtos`.

        This method inserts those formatoptions into `fmtos` that are required
        because one of the following criteria is fullfilled:

        1. The :attr:`replot` attribute is True
        2. Any formatoption with START priority is in `fmtos`
        3. A dependency of one formatoption is in `fmtos`

        Parameters
        ----------
        fmtos: list
            The list of formatoptions that shall be updated
        seen: set
            The formatoption keys that shall not be included. If None, all
            formatoptions in `fmtos` are used

        Returns
        -------
        fmtos
            The initial `fmtos` plus further formatoptions

        Notes
        -----
        `fmtos` and `seen` are modified in place (except that any formatoption
        in the initial `fmtos` has :attr:`~Formatoption.requires_clearing`
        attribute set to True)"""
        def get_dependencies(fmto):
            if fmto is None:
                return []
            return fmto.dependencies + list(chain(*map(
                lambda key: get_dependencies(getattr(self, key, None)),
                fmto.dependencies)))
        seen = seen or {fmto.key for fmto in fmtos}
        keys = {fmto.key for fmto in fmtos}
        if self.replot or any(fmto.priority >= START for fmto in fmtos):
            self.replot = True
            self.plot_data = self.data.copy(True)
            new_fmtos = dict((f.key, f) for f in self._fmtos
                             if ((f not in fmtos and is_data_dependent(
                                 f, self.data))))
            seen.update(new_fmtos)
            keys.update(new_fmtos)
            fmtos += list(new_fmtos.values())

        # insert the formatoptions that have to be updated if the plot is
        # changed
        if any(fmto.priority >= BEFOREPLOTTING for fmto in fmtos):
            new_fmtos = dict((f.key, f) for f in self._fmtos
                             if ((f not in fmtos and f.update_after_plot)))
            fmtos += list(new_fmtos.values())
        for fmto in set(self._fmtos).difference(fmtos):
            all_dependencies = get_dependencies(fmto)
            if keys.intersection(all_dependencies):
                fmtos.append(fmto)
            else:
                try:
                    fmto.lock.release()
                except RuntimeError:
                    pass
        if any(fmto.requires_clearing for fmto in fmtos):
            self.cleared = True
            return list(self._fmtos)
        return fmtos

    def _sorted_by_priority(self, fmtos, changed=None):
        """Sort the formatoption objects by their priority and dependency

        Parameters
        ----------
        fmtos: list
            list of :class:`Formatoption` instances
        changed: list
            the list of formatoption keys that have changed

        Yields
        ------
        Formatoption
            The next formatoption as it comes by the sorting

        Warnings
        --------
        The list `fmtos` is cleared by this method!"""
        def pop_fmto(key):
            idx = fmtos_keys.index(key)
            del fmtos_keys[idx]
            return fmtos.pop(idx)

        def get_children(fmto, parents_keys):
            all_fmtos = fmtos_keys + parents_keys
            for key in fmto.children + fmto.dependencies:
                if key not in fmtos_keys:
                    continue
                child_fmto = pop_fmto(key)
                for childs_child in get_children(
                        child_fmto, parents_keys + [child_fmto.key]):
                    yield childs_child
                # filter out if parent is in update list
                if (any(key in all_fmtos for key in child_fmto.parents) or
                        fmto.key in child_fmto.parents):
                    continue
                yield child_fmto

        fmtos.sort(key=lambda fmto: fmto.priority, reverse=True)
        fmtos_keys = [fmto.key for fmto in fmtos]
        self._last_update = changed or fmtos_keys[:]
        self.logger.debug("Update the formatoptions %s", fmtos_keys)
        while fmtos:
            del fmtos_keys[0]
            fmto = fmtos.pop(0)
            # first update children
            for child_fmto in get_children(fmto, [fmto.key]):
                yield child_fmto
            # filter out if parent is in update list
            if any(key in fmtos_keys for key in fmto.parents):
                continue
            yield fmto

    @classmethod
    def _get_formatoptions(cls, include_bases=True):
        """
        Iterator over formatoptions

        This class method returns an iterator that contains all the
        formatoptions descriptors that are in this class and that are defined
        in the base classes

        Notes
        -----
        There is absolutely no need to call this method besides the plotter
        initialization, since all formatoptions are in the plotter itself.
        Just type::

        >>> list(plotter)

        to get the formatoptions.

        See Also
        --------
        _format_keys"""
        def base_fmtos(base):
            return filter(
                lambda key: isinstance(getattr(cls, key), Formatoption),
                getattr(base, '_get_formatoptions', empty)(False))

        def empty(*args, **kwargs):
            return list()
        fmtos = (attr for attr, obj in six.iteritems(cls.__dict__)
                 if isinstance(obj, Formatoption))
        if not include_bases:
            return fmtos
        return unique_everseen(chain(fmtos, *map(base_fmtos, cls.__mro__)))

    docstrings.keep_types('check_key.parameters', 'kwargs',
                          '``\*args`` and ``\*\*kwargs``')

    @classmethod
    @docstrings.get_sectionsf('Plotter._enhance_keys')
    @docstrings.dedent
    def _enhance_keys(cls, keys=None, *args, **kwargs):
        """
        Enhance the given keys by groups

        Parameters
        ----------
        keys: list of str or None
            If None, the all formatoptions of the given class are used. Group
            names from the :attr:`psyplot.plotter.groups` mapping are replaced
            by the formatoptions

        Other Parameters
        ----------------
        %(check_key.parameters.kwargs)s

        Returns
        -------
        list of str
            The enhanced list of the formatoptions"""
        all_keys = list(cls._get_formatoptions())
        if isinstance(keys, six.string_types):
            keys = [keys]
        else:
            keys = list(keys or all_keys)
        fmto_groups = defaultdict(list)
        for key in all_keys:
            fmto_groups[getattr(cls, key).group].append(key)
        new_i = 0
        for i, key in enumerate(keys[:]):

            if key in fmto_groups:
                del keys[new_i]
                for key2 in fmto_groups[key]:
                    if key2 not in keys:
                        keys.insert(new_i, key2)
                        new_i += 1
            else:
                valid, similar, message = check_key(
                    key, all_keys, False, 'formatoption keyword', *args,
                    **kwargs)
                if not valid:
                    keys.remove(key)
                    warn(message)
            new_i += 1
        return keys

    @classmethod
    @docstrings.get_sectionsf('Plotter.show_keys',
                              sections=['Parameters', 'Returns'])
    @docstrings.dedent
    def show_keys(cls, keys=None, indent=0, grouped=False, func=six.print_,
                  *args, **kwargs):
        """
        Classmethod to return a nice looking table with the given formatoptions

        Parameters
        ----------
        %(Plotter._enhance_keys.parameters)s
        indent: int
            The indentation of the table
        grouped: bool, optional
            If True, the formatoptions are grouped corresponding to the
            :attr:`Formatoption.groupname` attribute
        func: function
            The function the is used for returning (by default it is printed
            via the :func:`print` function). It must take a string as argument
        %(Plotter._enhance_keys.other_parameters)s

        Returns
        -------
        results of `func`
            None if `func` is the print function, otherwise anything else

        See Also
        --------
        show_summaries, show_docs"""
        def titled_group(groupname):
            bars = str_indent + '*' * len(groupname) + '\n'
            return bars + str_indent + groupname + '\n' + bars

        # we use a local and global boolean here if the links shall be included
        # (see :attr:`_fmt_links` above) to make sure that the links are only
        # included if we really want it
        _this_fmt_links = kwargs.pop('_fmt_links', False)

        keys = cls._enhance_keys(keys, *args, **kwargs)
        str_indent = " " * indent
        if grouped:
            grouped_keys = defaultdict(list)
            for fmto in map(lambda key: getattr(cls, key), keys):
                grouped_keys[fmto.groupname].append(fmto.key)
            text = ""
            for group, keys in six.iteritems(grouped_keys):

                text += titled_group(group) + cls.show_keys(
                    keys, indent=indent, grouped=False, print_fmts=False) + \
                        '\n\n'
            return func(text.rstrip())
        if not keys:
            return
        ncols = min([4, len(keys)])  # number of columns
        if _this_fmt_links and _fmt_links:
            long_keys = list(map(lambda key: ':attr:`~%s.%s.%s`' % (
                cls.__module__, cls.__name__, key), keys))
        else:
            long_keys = keys
        maxn = max(map(len, long_keys))  # maximal lenght of the keys
        bars = str_indent + ("="*(maxn) + "  ")*ncols
        lines = (''.join(key.ljust(maxn + 2) for key in long_keys[i:i+ncols])
                 for i in range(0, len(keys), ncols))
        text = bars + "\n" + str_indent + ("\n" + str_indent).join(
            lines)
        if six.PY2:
            text = (text + "\n" + bars).encode('utf-8')
        else:
            text += "\n" + bars
        return func(text)

    @classmethod
    @docstrings.dedent
    def _show_doc(cls, fmt_func, keys=None, indent=0, grouped=False,
                  func=six.print_, *args, **kwargs):
        """
        Classmethod to print the formatoptions and their documentation

        This function is the basis for the :meth:`show_summaries` and
        :meth:`show_docs` methods

        Parameters
        ----------
        fmt_func: function
            A function that takes the documentation of a formatoption as
            argument and returns what shall be printed
        %(Plotter.show_keys.parameters)s

        Returns
        -------
        %(Plotter.show_keys.returns)s

        See Also
        --------
        show_summaries, show_docs"""
        def titled_group(groupname):
            bars = str_indent + '*' * len(groupname) + '\n'
            return bars + str_indent + groupname + '\n' + bars

        keys = cls._enhance_keys(keys, *args, **kwargs)
        str_indent = " " * indent
        if grouped:
            grouped_keys = defaultdict(list)
            for fmto in map(lambda key: getattr(cls, key), keys):
                grouped_keys[fmto.groupname].append(fmto.key)
            text = "\n\n".join(
                titled_group(group) + cls._show_doc(
                    fmt_func, keys, indent=indent, grouped=False,
                    func=str) for group, keys in six.iteritems(
                        grouped_keys))
            return func(text.rstrip())

        text = '\n'.join(str_indent + key + '\n' + fmt_func(
            getattr(cls, key).__doc__) for key in keys)
        return func(text)

    @classmethod
    @docstrings.dedent
    def show_summaries(cls, keys=None, indent=0, grouped=False,
                       func=six.print_, *args, **kwargs):
        """
        Classmethod to print the summaries of the formatoptions

        Parameters
        ----------
        %(Plotter.show_keys.parameters)s

        Returns
        -------
        %(Plotter.show_keys.returns)s

        See Also
        --------
        show_keys, show_docs"""
        def find_summary(doc):
            return '\n'.join(wrapper.wrap(doc[:doc.find('\n\n')]))
        str_indent = " " * indent
        wrapper = TextWrapper(width=80, initial_indent=str_indent + ' ' * 4,
                              subsequent_indent=str_indent + ' ' * 4)
        return cls._show_doc(find_summary, keys=keys, indent=indent,
                             grouped=grouped, func=func, *args,
                             **kwargs)

    @classmethod
    @docstrings.dedent
    def show_docs(cls, keys=None, indent=0, grouped=False, func=six.print_,
                  *args, **kwargs):
        """
        Classmethod to print the full documentations of the formatoptions

        Parameters
        ----------
        %(Plotter.show_keys.parameters)s

        Returns
        -------
        %(Plotter.show_keys.returns)s

        See Also
        --------
        show_keys, show_docs"""
        def full_doc(doc):
            return str_indent + ' ' * 4 + ('\n' + str_indent + ' ' * 4).join(
                doc.split('\n'))
        str_indent = " " * indent
        return cls._show_doc(full_doc, keys=keys, indent=indent,
                             grouped=grouped, func=func, *args, **kwargs)

    @classmethod
    def _get_rc_strings(cls):
        """
        Recursive method to get the base strings in the rcParams dictionary.

        This method takes the :attr:`_rcparams_string` attribute from the given
        `class` and combines it with the :attr:`_rcparams_string` attributes
        from the base classes.
        The returned frozenset can be used as base strings for the
        :meth:`psyplot.config.rcsetup.RcParams.find_and_replace` method.

        Returns
        -------
        list
            The first entry is the :attr:`_rcparams_string` of this class,
            the following the :attr:`_rcparams_string` attributes of the
            base classes according to the method resolution order of this
            class"""
        return list(unique_everseen(chain(
            *map(lambda base: getattr(base, '_rcparams_string', []),
                 cls.__mro__))))

    def _set_rc(self):
        """Method to set the rcparams and defaultParams for this plotter"""
        base_str = self._get_rc_strings()
        # to make sure that the '.' is not interpreted as a regex pattern,
        # we specify the pattern_base by ourselves
        pattern_base = map(lambda s: s.replace('.', '\.'), base_str)
        # pattern for valid keys being all formatoptions in this plotter
        pattern = '(%s)(?=$)' % '|'.join(self)
        self._rc = rcParams.find_and_replace(base_str, pattern=pattern,
                                             pattern_base=pattern_base)
        user_rc = SubDict(rcParams['plotter.user'], base_str, pattern=pattern,
                          pattern_base=pattern_base)
        self._rc.update(user_rc.data)

        self._defaultParams = SubDict(defaultParams, base_str, pattern=pattern,
                                      pattern_base=pattern_base)

    docstrings.keep_params('InteractiveBase.update.parameters', 'auto_update')

    @docstrings.dedent
    def update(self, fmt={}, replot=False, auto_update=False, draw=True,
               todefault=False, **kwargs):
        """
        Update the formatoptions and the plot

        If the :attr:`data` attribute of this plotter is None, the plotter is
        updated like a usual dictionary (see :meth:`dict.update`). Otherwise
        the update is registered and performed if `auto_update` is True or if
        the :meth:`start_update` method is called (see below).

        Parameters
        ----------
        %(Plotter._register_update.parameters)s
        %(InteractiveBase.start_update.parameters)s
        %(InteractiveBase.update.parameters.auto_update)s
        ``**kwargs``
            Any other formatoption that shall be updated (additionally to those
            in `fmt`)

        Notes
        -----
        %(InteractiveBase.update.notes)s"""
        if self.disabled:
            return
        fmt = dict(fmt)
        if kwargs:
            fmt.update(kwargs)
        # if the data is None, update like a usual dictionary (but with
        # validation)
        if self.data is None:
            for key, val in six.iteritems(fmt):
                self[key] = val

        self._register_update(fmt=fmt, replot=replot, todefault=todefault)
        if self.auto_update or auto_update:
            self.start_update(draw=draw)

    def _set_sharing_keys(self, keys):
        """
        Set the keys to share or unshare

        Parameters
        ----------
        keys: string or iterable of strings
            The iterable may contain formatoptions that shall be shared (or
            unshared), or group names of formatoptions to share all
            formatoptions of that group (see the :attr:`fmt_groups` property).
            If None, all formatoptions of this plotter are inserted.

        Returns
        -------
        set
            The set of formatoptions to share (or unshare)"""
        keys = set(self) if keys is None else set(keys)
        fmto_groups = self._fmto_groups
        keys.update(chain(*(map(lambda fmto: fmto.key, fmto_groups[key])
                            for key in keys.intersection(fmto_groups))))
        keys.difference_update(fmto_groups)
        return keys

    def share(self, plotters, keys=None, draw=True):
        """Share the formatoptions of this plotter with others

        This method shares the formatoptions of this :class:`Plotter` instance
        with others to make sure that, if the formatoption of this changes,
        those of the others change as well

        Parameters
        ----------
        plotters: list of :class:`Plotter` instances or a :class:`Plotter`
            The plotters to share the formatoptions with
        keys: string or iterable of strings
            The formatoptions to share, or group names of formatoptions to
            share all formatoptions of that group (see the
            :attr:`fmt_groups` property). If None, all formatoptions of this
            plotter are unshared.
        draw: bool
            If True, changed figures are drawn at the end

        See Also
        --------
        unshare, unshare_me"""
        if isinstance(plotters, Plotter):
            plotters = [plotters]
        keys = self._set_sharing_keys(keys)
        for plotter in plotters:
            for key in keys:
                plotter._shared[key] = getattr(self, key)
            for priority, grouper in plotter._grouped_fmtos(
                    plotter._sorted_by_priority(plotter._insert_additionals(
                        list(map(lambda key: getattr(plotter, key), keys))))):
                if plotter.cleared:
                    plotter.reinit()
                else:
                    plotter._plot_by_priority(priority, grouper)
                plotter.cleared = False
        if draw:
            self.draw()

    def unshare(self, plotters, keys=None):
        """Close the sharing connection of this plotter with others

        This method undoes the sharing connections made by the :meth:`share`
        method and releases the given `plotters` again, such that the
        formatoptions in this plotter may be updated again to values different
        from this one.

        Parameters
        ----------
        plotters: list of :class:`Plotter` instances or a :class:`Plotter`
            The plotters to release
        keys: string or iterable of strings
            The formatoptions to unshare, or group names of formatoptions to
            unshare all formatoptions of that group (see the
            :attr:`fmt_groups` property). If None, all formatoptions of this
            plotter are unshared.

        See Also
        --------
        share, unshare_me"""
        if isinstance(plotters, Plotter):
            plotters = [plotters]
        keys = self._set_sharing_keys(keys)
        for plotter in plotters:
            plotter.unshare_me(keys)

    def unshare_me(self, keys=None):
        """Close the sharing connection of this plotter with others

        This method undoes the sharing connections made by the :meth:`share`
        method and release this plotter again.

        Parameters
        ----------
        keys: string or iterable of strings
            The formatoptions to unshare, or group names of formatoptions to
            unshare all formatoptions of that group (see the
            :attr:`fmt_groups` property). If None, all formatoptions of this
            plotter are unshared.

        See Also
        --------
        share, unshare"""
        keys = self._set_sharing_keys(keys)
        for key in keys:
            fmto = getattr(self, key)
            try:
                other_fmto = self._shared.pop(key)
            except KeyError:
                pass
            else:
                other_fmto.shared.remove(fmto)
                fmto.update(fmto.value)

    def get_vfunc(self, key):
        """Return the validation function for a specified formatoption

        Parameters
        ----------
        key: str
            Formatoption key in the :attr:`rc` dictionary

        Returns
        -------
        function
            Validation function for this formatoption"""
        return self._defaultParams[key][1]

    def _save_state(self):
        """Saves the current formatoptions"""
        self._old_fmt.append(self.changed)

    @dedent
    def set_logger(self, name=None, force=False):
        """
        Sets the logging.Logger instance of this plotter.

        Parameters
        ----------
        name: str
            name of the Logger. If None and the :attr:`data` attribute is not
            None, it will be named like <module name>.<arr_name>.<class name>,
            where <arr_name> is the name of the array in the :attr:`data`
            attribute
        force: Bool.
            If False, do not set it if the instance has already a logger
            attribute."""
        import logging
        if name is None:
            try:
                name = '%s.%s.%s' % (self.__module__, self.data.arr_name,
                                     self.__class__.__name__)
            except AttributeError:
                name = '%s.%s' % (self.__module__, self.__class__.__name__)
        if not hasattr(self, 'logger') or force:
            self.logger = logging.getLogger(name)
            self.logger.debug('Initializing...')

    @dedent
    def has_changed(self, key, include_last=True):
        """
        Determine whether a formatoption changed in the last update

        Parameters
        ----------
        key: str
            A formatoption key contained in this plotter
        include_last: bool
            if True and the formatoption has been included in the last update,
            the return value will not be None. Otherwise the return value will
            only be not None if it changed during the last update

        Returns
        -------
        None or list
            - None, if the value has not been changed during the last update or
              `key` is not a valid formatoption key
            - a list of length two with the old value in the first place and
              the given `value` at the second"""
        if self._initializing or key not in self:
            return
        fmto = getattr(self, key)
        if self._old_fmt and key in self._old_fmt[-1]:
            old_val = self._old_fmt[-1][key]
        else:
            old_val = fmto.default
        if (fmto.diff(old_val) or (include_last and
                                   fmto.key in self._last_update)):
            return [old_val, fmto.value]

    def get_enhanced_attrs(self, arr, axes=['x', 'y', 't', 'z']):
        if isinstance(arr, InteractiveList):
            all_attrs = list(starmap(self.get_enhanced_attrs, zip(
                arr, repeat(axes))))
            attrs = {key: val for key, val in six.iteritems(all_attrs[0])
                     if all(key in attrs and attrs[key] == val
                            for attrs in all_attrs[1:])}
            attrs.update(arr.attrs)
        else:
            attrs = arr.attrs.copy()
            base_variables = self.base_variables
            if len(base_variables) > 1:  # multiple variables
                for name, base_var in six.iteritems(base_variables):
                    attrs.update(
                        {name+key: value for key, value in six.iteritems(
                            base_var.attrs)})
            else:
                base_var = next(six.itervalues(base_variables))
            attrs['name'] = arr.name
            for dim, coord in six.iteritems(arr.coords):
                if coord.size == 1:
                    attrs[dim] = format_time(coord.values)
            for dim in axes:
                if isinstance(self.data, InteractiveList):
                    decoder = self.data[0].decoder
                else:
                    decoder = self.data.decoder
                coord = getattr(decoder, 'get_' + dim)(
                    base_var, coords=arr.coords)
                if coord is None:
                    continue
                if coord.size == 1:
                    attrs[dim] = format_time(coord.values)
                attrs[dim + 'name'] = coord.name
                for key, val in six.iteritems(coord.attrs):
                    attrs[dim + key] = val
        self._enhanced_attrs = attrs
        return attrs

    def _make_plot(self):
        plotters = [fmto for fmto in self._fmtos if fmto.plot_fmt]
        plotters.sort(key=lambda fmto: fmto.priority, reverse=True)
        for fmto in plotters:
            self.logger.debug("Making plot with %s formatoption", fmto.key)
            fmto.make_plot()

    @classmethod
    def _get_sample_projection(cls):
        """Returns None. May be subclassed to return a projection that
        can be used when creating a subplot"""
        pass


def unique_everseen(iterable, key=None):
    """List unique elements, preserving order. Remember all elements ever seen.

    Function taken from https://docs.python.org/2/library/itertools.html"""
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element
