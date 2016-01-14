from threading import Thread
from glob import glob
import re
import six
from itertools import chain, product, repeat, starmap
import xray
from xray.core.utils import NDArrayMixin
from xray.core.formatting import first_n_items
from numpy import (
    array, unique, zeros, asarray, r_, arange, datetime64, dtype, vectorize,
    pi, reshape)
from datetime import datetime, timedelta
import logging
from .config.rcsetup import rcParams, safe_list
from .docstring import dedent, docstrings, dedents
from .compat.pycompat import zip, map, isstring, OrderedDict, filter, range
from .warning import warn, PsyPlotRuntimeWarning

if six.PY2:
    from Queue import Queue
else:
    from queue import Queue


# No data variable. This is used for filtering if an attribute could not have
# been accessed
_NODATA = object


logger = logging.getLogger(__name__)


def sort_kwargs(kwargs, *param_lists):
    """Function to sort keyword arguments and sort them into dictionaries

    This function returns dictionaries that contain the keyword arguments
    from `kwargs` corresponding given iterables in ``*params``

    Parameters
    ----------
    kwargs: dict
        Original dictionary
    ``*param_lists``
        iterables of strings, each standing for a possible key in kwargs

    Returns
    -------
    list
        len(params) + 1 dictionaries. Each dictionary contains the items of
        `kwargs` corresponding to the specified list in ``*param_lists``. The
        last dictionary contains the remaining items"""
    return chain(
        ({key: kwargs.pop(key) for key in params.intersection(kwargs)}
         for params in map(set, param_lists)), [kwargs])


def _infer_interval_breaks(coord):
    """
    >>> _infer_interval_breaks(np.arange(5))
    array([-0.5,  0.5,  1.5,  2.5,  3.5,  4.5])

    Taken from xray.plotting.plot module
    """
    coord = asarray(coord)
    deltas = 0.5 * (coord[1:] - coord[:-1])
    first = coord[0] - deltas[0]
    last = coord[-1] + deltas[-1]
    return r_[[first], coord[:-1] + deltas, [last]]


@docstrings.get_sectionsf('setup_coords')
@dedent
def setup_coords(arr_names=None, sort=[], dims={}, **kwargs):
    """
    Sets up the arr_names dictionary for the plot

    Parameters
    ----------
    arr_names: string, list of strings or dictionary
        Set the unique array names of the resulting arrays and (optionally)
        dimensions.

        - if string: same as list of strings (see below). Strings may
          include {0} which will be replaced by a counter.
        - list of strings: those will be used for the array names. The final
          number of dictionaries in the return depend in this case on the
          `dims` and ``**furtherdims``
        - dictionary:
          Then nothing happens and an :class:`OrderedDict` version of
          `arr_names` is returned.
    sort: list of strings
        This parameter defines how the dictionaries are ordered. It has no
        effect if `arr_names` is a dictionary (use a
        :class:`~collections.OrderedDict` for that). It can be a list of
        dimension strings matching to the dimensions in `dims` for the
        variable.
    dims: dict
        Keys must be variable names of dimensions (e.g. time, level, lat or
        lon) or 'name' for the variable name you want to choose.
        Values must be values of that dimension or iterables of the values
        (e.g. lists). Note that strings will be put into a list.
        For example dims = {'name': 't2m', 'time': 0} will result in one plot
        for the first time step, whereas dims = {'name': 't2m', 'time': [0, 1]}
        will result in two plots, one for the first (time == 0) and one for the
        second (time == 1) time step.
    ``**kwargs``
        The same as `dims` (those will update what is specified in `dims`)

    Returns
    -------
    ~collections.OrderedDict
        A mapping from the keys in `arr_names` and to dictionaries. Each
        dictionary corresponds defines the coordinates of one data array to
        load"""
    try:
        return OrderedDict(arr_names)
    except TypeError:
        pass
    if arr_names is None:
        arr_names = repeat('arr{0}')
    dims = OrderedDict(dims)
    for key, val in six.iteritems(kwargs):
        dims[key] = val
    sorted_dims = OrderedDict()
    if sort:
        for key in sort:
            sorted_dims[key] = dims.pop(key)
        for key, val in six.iteritems(dims):
            sorted_dims[key] = val
    else:
        sorted_dims = dims
    for key, val in six.iteritems(sorted_dims):
        sorted_dims[key] = iter(safe_list(val))
    return OrderedDict([
        (arr_name.format(i), dict(zip(sorted_dims.keys(), dim_tuple)))
        for i, (arr_name, dim_tuple) in enumerate(zip(
            arr_names, product(
                *map(list, sorted_dims.values()))))])


def is_slice(arr):
    """Test whether `arr` is an integer array that can be replaced by a slice

    Parameters
    ----------
    arr: numpy.array
        Numpy integer array

    Returns
    -------
    slice or None
        If `arr` could be converted to an array, this is returned, otherwise
        `None` is returned

    See Also
    --------
    get_index_from_coord"""
    if len(arr) == 1:
        return slice(arr[0], arr[0] + 1)
    step = unique(arr[1:] - arr[:-1])
    if len(step) == 1:
        return slice(arr[0], arr[-1], step)


def get_index_from_coord(coord, base_index):
    """Function to return the coordinate as integer, integer array or slice

    If `coord` is zero-dimensional, the corresponding integer in `base_index`
    will be supplied. Otherwise it is first tried to return a slice, if that
    does not work an integer array with the corresponding indices is returned.

    Parameters
    ----------
    coord: xray.Coordinate or xray.Variable
        Coordinate to convert
    base_index: pandas.Index
        The base index from which the `coord` was extracted

    Returns
    -------
    int, array of ints or slice
        The indexer that can be used to access the `coord` in the
        `base_index`
    """
    try:
        values = coord.values
    except AttributeError:
        values = coord
    if values.ndim == 0:
        return base_index.get_loc(values[()])
    if len(values) == len(base_index) and (values == base_index).all():
        return slice(None)
    values = array(list(map(lambda i: base_index.get_loc(i), values)))
    return is_slice(values) or values


class InteractiveBase(object):
    """Class for the communication of a data object with a suitable plotter

    This class serves as an interface for data objects (in particular as a
    base for :class:`InteractiveArray` and :class:`InteractiveList`) to
    communicate with the corresponding :class:`~psyplot.plotter.Plotter` in the
    :attr:`plotter` attribute"""
    @property
    def plotter(self):
        """:class:`psyplot.plotter.Plotter` instance that makes the interactive
        plotting of the data"""
        return self._plotter

    @plotter.setter
    def plotter(self, value):
        self._plotter = value

    @plotter.deleter
    def plotter(self):
        self._plotter = None

    @property
    def auto_update(self):
        """:class:`bool`. Boolean controlling whether the :meth:`start_update`
        method is automatically called by the :meth:`update` method"""
        return self._auto_update

    @auto_update.setter
    def auto_update(self, value):
        if self.plotter is not None:
            self.plotter.auto_update = value
        self._auto_update = value

    _plotter = None

    @property
    @docstrings.save_docstring('InteractiveBase._njobs')
    @dedent
    def _njobs(self):
        """
        The number of jobs taken from the queue during an update process

        Returns
        -------
        list of int
            The length of the list determines the number of neccessary queues,
            the numbers in the list determines the number of tasks per queue
            this instance fullfills during the update process"""
        return self.plotter._njobs if self.plotter is not None else []

    #: :class:`str`. The internal name of the :class:`InteractiveBase` instance
    arr_name = None

    @docstrings.get_sectionsf('InteractiveBase')
    @dedent
    def __init__(self, plotter=None, arr_name='data', auto_update=None):
        """
        Parameters
        ----------
        plotter: Plotter
            Default: None. Interactive plotter that makes the plot via
            formatoption keywords.
        arr_name: str
            Default: ``'data'``. unique string of the array
        auto_update: bool
            Default: None. A boolean indicating whether this list shall
            automatically update the contained arrays when calling the
            :meth:`update` method or not. See also the :attr:`auto_update`
            attribute. If None, the value from the ``lists.auto_update'``
            key in the :attr:`psyplot.rcParams` dictionary is used."""
        self.plotter = plotter
        self.arr_name = arr_name
        if auto_update is None:
            auto_update = rcParams['lists.auto_update']
        self.auto_update = auto_update
        self.replot = False

    def interactive_plot(self, plotter=None):
        """Makes the interactive plot

        Parameters
        ----------
        plotter: Plotter
            Interactive plotter that makes the plot via formatoption keywords.
            If None, whatever is found in the :attr:`plotter` attribute is
            used.
        """
        self.plotter = plotter or self.plotter
        if self.plotter is None:
            raise ValueError(
                "Found no plotter in the InteractiveArray instance!")
        self.plotter.initialize_plot(self)

    @docstrings.get_sectionsf('InteractiveBase._register_update')
    @dedent
    def _register_update(self, replot=False, fmt={}, force=False,
                         todefault=False):
        """
        Register new formatoptions for updating

        Parameters
        ----------
        replot: bool
            Boolean that determines whether the data specific formatoptions
            shall be updated in any case or not. Note, if `dims` is not empty
            or any coordinate keyword is in ``**kwargs``, this will be set to
            True automatically
        fmt: dict
            Keys may be any valid formatoption of the formatoptions in the
            :attr:`plotter`
        force: str, list of str or bool
            If formatoption key (i.e. string) or list of formatoption keys,
            thery are definitely updated whether they changed or not.
            If True, all the given formatoptions in this call of the are
            :meth:`update` method are updated
        todefault: bool
            If True, all changed formatoptions (except the registered ones)
            are updated to their default value as stored in the
            :attr:`~psyplot.plotter.Plotter.rc` attribute

        See Also
        --------
        start_update"""
        self.replot = self.replot or replot
        if self.plotter is not None:
            self.plotter._register_update(replot=self.replot, fmt=fmt,
                                          force=force, todefault=todefault)

    @docstrings.get_sectionsf('InteractiveBase.start_update',
                              sections=['Parameters', 'Returns'])
    @dedent
    def start_update(self, draw=True, queues=None):
        """
        Conduct the formerly registered updates

        This method conducts the updates that have been registered via the
        :meth:`update` method. You can call this method if the
        :attr:`auto_update` attribute of this instance and the `auto_update`
        parameter in the :meth:`update` method has been set to False

        Parameters
        ----------
        draw: bool
            Boolean to control whether the figure of this array shall be drawn
            at the end
        queues: list of :class:`Queue.Queue` instances
            The queues that are passed to the
            :meth:`psyplot.plotter.Plotter.start_update` method to ensure a
            thread-safe update. It can be None if only one single plotter is
            updated at the same time. The number of jobs that are taken from
            the queue is determined by the :meth:`_njobs` attribute. Note that
            there this parameter is automatically configured when updating
            from a :class:`~psyplot.project.Project`.

        Returns
        -------
        bool
            A boolean indicating whether a redrawing is necessary or not

        See Also
        --------
        :attr:`auto_update`, update
        """
        if self.plotter is not None:
            return self.plotter.start_update(draw=draw, queues=queues)

    docstrings.keep_params('InteractiveBase.start_update.parameters', 'draw')

    @docstrings.get_sectionsf('InteractiveBase.update',
                              sections=['Parameters', 'Notes'])
    @docstrings.dedent
    def update(self, fmt={}, replot=False, draw=True, auto_update=False,
               force=False, todefault=False, **kwargs):
        """
        Update the coordinates and the plot

        This method updates all arrays in this list with the given coordinate
        values and formatoptions.

        Parameters
        ----------
        %(InteractiveBase._register_update.parameters)s
        auto_update: bool
            Boolean determining whether or not the :meth:`start_update` method
            is called at the end.
        %(InteractiveBase.start_update.parameters.draw)s
        ``**kwargs``
            Any other formatoption that shall be updated (additionally to those
            in `fmt`)

        Notes
        -----
        If the :attr:`auto_update` attribute and the given `auto_update`
        parameter are both False, the update of the plots are registered and
        conducted at the next call of the :meth:`start_update` method or the
        next call of this method (if the `auto_update` parameter is then True).
        """
        fmt = dict(fmt)
        fmt.update(kwargs)

        self._register_update(replot=replot, fmt=fmt, force=force,
                              todefault=todefault)

        if self.auto_update or auto_update:
            self.start_update(draw=draw)


class InteractiveArray(xray.DataArray, InteractiveBase):
    """Interactive version of the :class:`xray.DataArray`

    This class keeps reference to the base :class:`xray.Dataset` where it
    originates from and enables to switch between the coordinates in this
    array. Furthermore it has a :attr:`plotter` attribute to enable interactive
    plotting via an :class:`psyplot.plotter.Plotter` instance."""

    @property
    def base(self):
        """Base dataset this instance gets its data from"""
        return getattr(self, '_base', self.to_dataset())

    @base.setter
    def base(self, value):
        self._base = value

    @property
    def decoder(self):
        """The decoder of this array"""
        return getattr(self, '_decoder', CFDecoder(self.base))

    @decoder.setter
    def decoder(self, value):
        self._decoder = value

    @property
    def idims(self):
        """Coordinates in the :attr:`base` dataset as int or slice

        This attribute holds a mapping from the coordinate names of this
        array to an integer, slice or an array of integer that represent the
        coordinates in the :attr:`base` dataset"""
        if self._idims is None:
            self._idims = self.decoder.get_idims(self)
        return self._idims

    @idims.setter
    def idims(self, value):
        self._idims = value

    @property
    @docstrings
    def _njobs(self):
        """%(InteractiveBase._njobs)s"""
        ret = super(self.__class__, self)._njobs or [0]
        ret[0] += 1
        return ret

    @docstrings.dedent
    def __init__(self, *args, **kwargs):
        """
        The ``*args`` and ``**kwargs`` are essentially the same as for the
        :class:`xray.DataArray` method, additional ``**kwargs`` are
        described below.

        Other Parameters
        ----------------
        base: xray.Dataset
            Default: None. Dataset that serves as the origin of the data
            contained in this DataArray instance. This will be used if you want
            to update the coordinates via the :meth:`update` method. If None,
            this instance will serve as a base as soon as it is needed.
        decoder: psyplot.CFDecoder
            The decoder that decodes the `base` dataset and is used to get
            bounds. If not given, a new :class:`CFDecoder` is created
        idims: dict
            Default: None. dictionary with integer values and/or slices in the
            `base` dictionary. If not given, they are determined automatically
        %(InteractiveBase.parameters)s
        """
        base = kwargs.pop('base', None)
        if base is not None:
            self.base = base
        self.idims = kwargs.pop('idims', None)
        decoder = kwargs.pop('decoder', None)
        if decoder is not None:
            self.decoder = decoder

        ibase_kwargs, array_kwargs = sort_kwargs(
            kwargs, ['plotter', 'arr_name', 'auto_update'])
        self._registered_updates = {}
        self._new_dims = {}
        self.method = None
        InteractiveBase.__init__(self, **ibase_kwargs)
        xray.DataArray.__init__(self, *args, **kwargs)

    @classmethod
    def _new_from_dataset_no_copy(cls, *args, **kwargs):
        obj = super(cls, cls)._new_from_dataset_no_copy(*args, **kwargs)
        obj._registered_updates = {}
        obj._new_dims = {}
        obj.method = None
        obj.arr_name = 'arr'
        obj.auto_update = kwargs.pop('auto_update',
                                     rcParams['lists.auto_update'])
        obj.replot = False
        return obj

    def copy(self, *args, **kwargs):
        arr_name = kwargs.pop('arr_name', self.arr_name)
        obj = super(InteractiveArray, self).copy(*args, **kwargs)
        obj.auto_update = self.auto_update
        obj.arr_name = arr_name
        obj.replot = self.replot
        return obj

    copy.__doc__ = xray.DataArray.copy.__doc__ + """
    Parameters
    ----------
    deep: bool
        If True, a deep copy is made of all variables in the underlying
        dataset. Otherwise, a shallow copy is made, so each variable in the new
        array's dataset is also a variable in this array's dataset.
    arr_name: str
        The array name to use for the new :class:`InteractiveArray` instance
        (default: arr)"""

    @property
    def base_variables(self):
        """A mapping from the variable name to the variablein the :attr:`base`
        dataset."""
        if 'variable' in self.coords:
            return OrderedDict([(name, self.base.variables[name])
                                for name in self.coords['variable'].values])
        return {self.name: self.base.variables[self.name]}

    docstrings.keep_params('setup_coords.parameters', 'dims')

    @docstrings.get_sectionsf('InteractiveArray._register_update')
    @docstrings.dedent
    def _register_update(self, method='isel', replot=False, dims={}, fmt={},
                         force=False, todefault=False):
        """
        Register new dimensions and formatoptions for updating

        Parameters
        ----------
        method: {'isel', None, 'nearest', ...}
            Selection method of the xray.Dataset to be used for setting the
            variables from the informations in `dims`.
            If `method` is 'isel', the :meth:`xray.Dataset.isel` method is
            used. Otherwise it sets the `method` parameter for the
            :meth:`xray.Dataset.sel` method.
        %(setup_coords.parameters.dims)s
        %(InteractiveBase._register_update.parameters)s

        See Also
        --------
        start_update"""
        if self._new_dims and self.method != method:
            raise ValueError(
                "New dimensions were already specified for with the %s method!"
                " I can not choose a new method %s" % (self.method, method))
        else:
            self.method = method
        self._new_dims.update(self.decoder.correct_dims(
            next(six.itervalues(self.base_variables)), dims))
        InteractiveBase._register_update(self, fmt=fmt, replot=replot or dims,
                                         force=force, todefault=todefault)

    def _update_concatenated(self, dims, method):
        """Updates a concatenated array to new dimensions"""
        def filter_attrs(item):
            """Checks whether the attribute is from the :attr:`base` dataset"""
            return (item[0] not in self.base.attrs or
                    item[1] != self.base.attrs[item[0]])
        saved_attrs = list(filter(filter_attrs, six.iteritems(self.attrs)))
        saved_name = self.name
        self.name = None
        if 'name' in dims:
            name = dims.pop('name')
        else:
            name = list(self.coords['variable'].values)
        if method == 'isel':
            self.idims.update(dims)
            dims = self.idims
            self._dataset = self.base[name].isel(
                **dims).to_array().to_dataset()
        else:
            for key, val in six.iteritems(self.coords):
                dims.setdefault(key, val)
            self._dataset = self.base[name].sel(method=method,
                                                **dims).to_array().to_dataset()
        self.name = saved_name
        for key, val in saved_attrs:
            self.attrs[key] = val

    def _update_array(self, dims, method):
        """Updates the array to the new dims from then :attr:`base` dataset"""
        def filter_attrs(item):
            """Checks whether the attribute is from the base variable"""
            return ((item[0] not in base_var.attrs or
                     item[1] != base_var.attrs[item[0]]))
        base_var = self.base.variables[self.name]
        if 'name' in dims:
            name = dims.pop('name')
            self.name = name
        else:
            name = self.name
        # save attributes that have been changed by the user
        saved_attrs = list(filter(filter_attrs, six.iteritems(self.attrs)))
        if method == 'isel':
            self.idims.update(dims)
            dims = self.idims
            self._dataset = self.base[name].isel(**dims).to_dataset()
        else:
            for key, val in six.iteritems(self.coords):
                dims.setdefault(key, val)
            self._dataset = self.base[name].sel(method=method,
                                                **dims).to_dataset()
        # update to old attributes
        for key, val in saved_attrs:
            self.attrs[key] = val

    @docstrings.dedent
    def start_update(self, draw=True, queues=None):
        """
        Conduct the formerly registered updates

        This method conducts the updates that have been registered via the
        :meth:`update` method. You can call this method if the
        :attr:`auto_update` attribute of this instance and the `auto_update`
        parameter in the :meth:`update` method has been set to False

        Parameters
        ----------
        %(InteractiveBase.start_update.parameters)s

        Returns
        -------
        %(InteractiveBase.start_update.returns)s

        See Also
        --------
        :attr:`auto_update`, update
        """
        def filter_attrs(item):
            return (item[0] not in self.base.attrs or
                    item[1] != self.base.attrs[item[0]])
        if queues is not None:
            # make sure that no plot is updated during gathering the data
            queues[0].get()
        dims = self._new_dims
        method = self.method
        if dims:
            if 'variable' in self.coords:
                self._update_concatenated(dims, method)
            else:
                self._update_array(dims, method)
        if queues is not None:
            queues[0].task_done()
        self._new_dims = {}
        return InteractiveBase.start_update(self, draw=draw, queues=queues)

    @docstrings.get_sectionsf('InteractiveArray.update',
                              sections=['Parameters', 'Notes'])
    @docstrings.dedent
    def update(self, method='isel', dims={}, fmt={}, replot=False,
               auto_update=False, draw=True, force=False, todefault=False,
               **kwargs):
        """
        Update the coordinates and the plot

        This method updates all arrays in this list with the given coordinate
        values and formatoptions.

        Parameters
        ----------
        %(InteractiveArray._register_update.parameters)s
        auto_update: bool
            Boolean determining whether or not the :meth:`start_update` method
            is called after the end.
        %(InteractiveBase.start_update.parameters)s
        ``**kwargs``
            Any other formatoption or dimension that shall be updated
            (additionally to those in `fmt` and `dims`)

        Notes
        -----
        %(InteractiveBase.update.notes)s"""
        dims = dict(dims)
        fmt = dict(fmt)
        vars_and_coords = set(chain(
            self.dims, self.coords, ['name', 'x', 'y', 'z', 'z']))
        furtherdims, furtherfmt = sort_kwargs(kwargs, vars_and_coords)
        dims.update(furtherdims)
        fmt.update(furtherfmt)

        self._register_update(method=method, replot=replot, dims=dims,
                              fmt=fmt, force=force, todefault=todefault)

        if self.auto_update or auto_update:
            self.start_update(draw=draw)


class ArrayList(list):
    """Base class for creating a list of interactive arrays from a dataset

    This list contains and manages :class:`InteractiveArray` instances"""

    docstrings.keep_params('InteractiveBase.parameters', 'auto_update')

    @property
    def dims(self):
        """Dimensions of the arrays in this list"""
        return set(chain(*(arr.dims for arr in self)))

    @property
    def arr_names(self):
        """Names of the arrays (!not of the variables!) in this list"""
        return list(arr.arr_name for arr in self)

    @property
    def names(self):
        """Set of the variable in this list"""
        return set(arr.name for arr in self)

    @property
    def coords(self):
        """Names of the coordinates of the arrays in this list"""
        return set(chain(*(arr.coords for arr in self)))

    @property
    def auto_update(self):
        """:class:`bool`. Boolean controlling whether the :meth:`start_update`
        method is automatically called by the :meth:`update` method"""
        return self._auto_update

    @auto_update.setter
    def auto_update(self, value):
        for arr in self:
            arr.auto_update = value
        self._auto_update = value

    docstrings.keep_params('InteractiveBase.parameters', 'auto_update')

    @docstrings.get_sectionsf('ArrayList')
    @docstrings.dedent
    def __init__(self, iterable=[], attrs={}, auto_update=None):
        """
        Parameters
        ----------
        iterable: iterable
            The iterable (e.g. another list) defining this list
        attrs: dict-like or iterable, optional
            Global attributes of this list
        %(InteractiveBase.parameters.auto_update)s"""
        super(ArrayList, self).__init__((arr for arr in iterable
                                         if isinstance(arr, InteractiveBase)))
        self.attrs = OrderedDict(attrs)
        if auto_update is None:
            auto_update = rcParams['lists.auto_update']
        self.auto_update = auto_update

    def copy(self, deep=False):
        """Returns a copy of the list

        Parameters
        ----------
        deep: bool
            If False (default), only the list is copied and not the contained
            arrays, otherwise the contained arrays are deep copied"""
        if not deep:
            return self.__class__(self[:], attrs=self.attrs.copy(),
                                  auto_update=self.auto_update)
        else:
            return self.__class__(
                [arr.copy(deep) for arr in self], attrs=self.attrs.copy(),
                auto_update=self.auto_update)

    docstrings.keep_params('InteractiveArray.update.parameters', 'method')

    @classmethod
    @docstrings.get_sectionsf('ArrayList.from_dataset', sections=[
        'Parameters', 'Other Parameters', 'Returns'])
    @docstrings.dedent
    def from_dataset(cls, base, method='isel', default_slice=None,
                     decoder=None, auto_update=None, prefer_list=False,
                     **kwargs):
        """
        Construct an ArrayDict instance from an existing base dataset

        Parameters
        ----------
        base: xray.Dataset
            Dataset instance that is used as reference
        %(InteractiveArray.update.parameters.method)s
        %(InteractiveBase.parameters.auto_update)s
        prefer_list: bool
            If True and multiple variable names per array are found, the
            :class:`InteractiveList` class is used. Otherwise the arrays are
            put together into one :class:`InteractiveArray`.
        default_slice: indexer
            Index (e.g. 0 if `method` is 'isel') that shall be used for
            dimensions not covered by `dims` and `furtherdims`. If None, the
            whole slice will be used.
        decoder: CFDecoder
            The decoder that shall be used to decoder the `base` dataset

        Other Parameters
        ----------------
        %(setup_coords.parameters)s

        Returns
        -------
        ArrayList
            The list with the specified :class:`InteractiveArray` instances
            that hold a reference to the given `base`"""
        def recursive_selection(key, dims, names):
            names = safe_list(names)
            if len(names) > 1 and prefer_list:
                keys = ('-'.join(vlst) for vlst in map(
                    safe_list, names))
                return InteractiveList(starmap(
                    sel_method, zip(keys, repeat(dims), names)),
                    auto_update=auto_update, arr_name=key)
            elif len(names) > 1:
                return sel_method(key, dims, tuple(names))
            else:
                return sel_method(key, dims, names[0])
        if method == 'isel':
            def sel_method(key, dims, name=None):
                if name is None:
                    return recursive_selection(key, dims, dims.pop('name'))
                elif isinstance(name, six.string_types):
                    arr = base[name]
                else:
                    arr = base[list(name)]
                if not isinstance(arr, xray.DataArray):
                    attrs = next(var for key, var in arr.variables.items()
                                 if key not in arr.coords).attrs
                    arr = arr.to_array()
                    arr.attrs.update(attrs)
                def_slice = slice(None) if default_slice is None else \
                    default_slice
                dims = decoder.correct_dims(arr, dims)
                dims.update({
                    dim: def_slice for dim in set(arr.dims).difference(
                        dims) if dim != 'variable'})
                return InteractiveArray(arr.isel(**dims), arr_name=key,
                                        base=base, idims=dims)
        else:
            def sel_method(key, dims, name=None):
                if name is None:
                    return recursive_selection(key, dims, dims.pop('name'))
                arr = base[name]
                if not isinstance(arr, xray.DataArray):
                    attrs = next(var for key, var in arr.variables.items()
                                 if key not in arr.coords).attrs
                    arr = arr.to_array()
                    arr.attrs.update(attrs)
                # idims will be calculated by the array (maybe not the most
                # efficient way...)
                dims = decoder.correct_dims(arr, dims)
                if default_slice is not None:
                    dims.update({
                        key: default_slice for key in set(arr.dims).difference(
                            dims)})
                return InteractiveArray(
                    arr.sel(method=method, **dims), arr_name=key, base=base)
        kwargs.setdefault(
            'name', sorted(
                key for key in base.variables if key not in base.coords))
        names = setup_coords(**kwargs)
        decoder = decoder or CFDecoder(base)
        instance = cls(starmap(sel_method, six.iteritems(names)),
                       attrs=base.attrs, auto_update=auto_update)
        # convert to interactive lists if an instance is not
        if prefer_list and any(
                not isinstance(arr, InteractiveList) for arr in instance):
            # if any instance is an interactive list, than convert the others
            if any(isinstance(arr, InteractiveList) for arr in instance):
                for i, arr in enumerate(instance):
                    if not isinstance(arr, InteractiveList):
                        instance[i] = InteractiveList([arr])
            else:  # put everything into one single interactive list
                instance = cls([InteractiveList(instance, attrs=base.attrs,
                                                auto_update=auto_update)])
        return instance

    @docstrings.dedent
    def _register_update(self, method='isel', replot=False, dims={}, fmt={},
                         force=False, todefault=False):
        """
        Register new dimensions and formatoptions for updating. The keywords
        are the same as for each single array

        Parameters
        ----------
        %(InteractiveArray._register_update.parameters)s"""

        for arr in self:
            arr._register_update(method=method, replot=replot, dims=dims,
                                 fmt=fmt, force=force, todefault=todefault)

    @docstrings.get_sectionsf('ArrayList.start_update')
    @dedent
    def start_update(self, draw=True):
        """
        Conduct the registered plot updates

        This method starts the updates from what has been registered by the
        :meth:`update` method. You can call this method if you did not set the
        `auto_update` parameter when calling the :meth:`update` method and when
        the :attr:`auto_update` attribute is False.

        Parameters
        ----------
        draw: bool
            If True, all the figures of the arrays contained in this list will
            be drawn at the end.

        See Also
        --------
        :attr:`auto_update`, update"""
        def worker(arr):
            results[arr.arr_name] = arr.start_update(draw=False, queues=queues)
        results = {}
        threads = [Thread(target=worker, args=(arr,),
                          name='update_%s' % arr.arr_name)
                   for arr in self]
        jobs = [arr._njobs for arr in self]
        queues = [Queue() for _ in range(max(map(len, jobs)))]
        # populate the queues
        for i, arr in enumerate(self):
            for j, n in enumerate(jobs[i]):
                for k in range(n):
                    queues[j].put(arr.arr_name)
        try:
            for thread in threads:
                thread.setDaemon(True)
                thread.start()
            for thread in threads:
                thread.join()
        except:
            raise

        if draw:
            self(arr_name=[name for name, adraw in six.iteritems(results)
                           if adraw]).draw()

    docstrings.keep_params('InteractiveArray.update.parameters',
                           'auto_update')

    @docstrings.get_sectionsf('ArrayList.update')
    @docstrings.dedent
    def update(self, method='isel', dims={}, fmt={}, replot=False,
               auto_update=False, draw=True, force=False, todefault=False,
               **kwargs):
        """
        Update the coordinates and the plot

        This method updates all arrays in this list with the given coordinate
        values and formatoptions.

        Parameters
        ----------
        %(InteractiveArray._register_update.parameters)s
        %(InteractiveArray.update.parameters.auto_update)s
        %(ArrayList.start_update.parameters)s
        ``**kwargs``
            Any other formatoption or dimension that shall be updated
            (additionally to those in `fmt` and `dims`)

        Notes
        -----
        %(InteractiveArray.update.notes)s

        See Also
        --------
        auto_update, start_update"""
        dims = dict(dims)
        fmt = dict(fmt)
        vars_and_coords = set(chain(
            self.dims, self.coords, ['name', 'x', 'y', 'z', 't']))
        furtherdims, furtherfmt = sort_kwargs(kwargs, vars_and_coords)
        dims.update(furtherdims)
        fmt.update(furtherfmt)

        self._register_update(method=method, replot=replot, dims=dims, fmt=fmt,
                              force=force, todefault=todefault)
        if self.auto_update or auto_update:
            self.start_update(draw)

    def draw(self):
        """Draws all the figures in this instance"""
        for fig in set(chain(*map(lambda arr: arr.plotter.figs2draw, self))):
            fig.canvas.draw()
        for arr in self:
            arr.plotter._figs2draw.clear()

    def __call__(self, types=None, **attrs):
        """Get the arrays specified by their attributes

        Parameters
        ----------
        types: type or tuple of types
            Any class that shall be used for an instance check via
            :func:`isinstance`. If not None, the :attr:`plotter` attribute
            of the array is checked against this `types`
        ``**attrs``
            Parameters may be any attribute of the arrays in this instance.
            Values may be iterables (e.g. lists) of the attributes to consider.
            If the value is a string, it will be put into a list."""
        def safe_item_list(key, val):
            return key, safe_list(val)
        attrs = list(starmap(safe_item_list, six.iteritems(attrs)))
        return self.__class__(
            # iterable
            (arr for arr in self if
             (types is None or isinstance(arr.plotter, types)) and
             (not attrs or
              all(getattr(arr, key, _NODATA) in val for key, val in attrs))),
            # give itself as base and the auto_update parameter
            auto_update=self.auto_update)

    def __contains__(self, val):
        try:
            name = val if isstring(val) else val.arr_name
        except AttributeError:
            raise ValueError(
                "Only interactive arrays can be inserted in the %s" % (
                    self.__class__.__name__))
        else:
            return name in self.arr_names and (
                isstring(val) or self._contains_array(val))

    def _contains_array(self, val):
        """Checks whether exactly this array is in the list"""
        if val.arr_name not in self.arr_names:
            return False
        arr = [arr for arr in self if arr.arr_name == val.arr_name][0]
        is_not_list = any(
            map(lambda a: not isinstance(a, InteractiveList),
                [arr, val]))
        is_list = any(map(lambda a: isinstance(a, InteractiveList),
                          [arr, val]))
        # if one is an InteractiveList and the other not, they differ
        if is_list and is_not_list:
            return False
        # if both are interactive lists, check the lists
        if is_list:
            return all(a in arr for a in val)
        # else we check the shapes and values
        return arr.shape == val.shape and (
            arr.values == val.values).all()

    @docstrings.get_sectionsf('ArrayList.rename', sections=[
        'Parameters', 'Raises'])
    @dedent
    def rename(self, arr, new_name=None):
        """
        Rename an array to find a name that isn't already in the list

        Parameters
        ----------
        arr: InteractiveBase
            A :class:`InteractiveArray` or :class:`InteractiveList` instance
            whose name shall be checked
        new_name: str
            If False, and the ``arr_name`` attribute of the new array is
            already in the list, a ValueError is raised.
            If None and the ``arr_name`` attribute of the new array is not
            already in the list, the name is not changed. Otherwise, if the
            array name is already in use, `new_name` is set to 'arr{0}'.
            If not None, this will be used for renaming (if the array name of
            `arr` is in use or not). ``'{0}'`` is replaced by a counter

        Returns
        -------
        InteractiveBase
            `arr` with changed ``arr_name`` attribute
        bool or None
            True, if the array has been renamed, False if not and None if the
            array is already in the list

        Raises
        ------
        ValueError
            If it was impossible to find a name that isn't already  in the list
        ValueError
            If `new_name` is False and the array is already in the list"""
        name_in_me = arr.arr_name in self.arr_names
        if new_name is not None or name_in_me:
            in_me = self._contains_array(arr)
            if new_name is False and name_in_me and not in_me:
                raise ValueError(
                    "Array name %s is already in use! Set the `new_name` "
                    "parameter to None for renaming!" % arr.arr_name)
            elif in_me:
                return arr, None
            elif new_name is False:
                return arr, False
            else:  # rename the array
                new_name = new_name if isstring(new_name) else 'arr{0}'
                names = self.arr_names
                try:
                    arr.arr_name = next(
                        filter(lambda n: n not in names,
                               map(new_name.format, range(100))))
                except StopIteration:
                    raise ValueError(
                        "{0} already in the list".format(new_name))
                return arr, False
        return arr, False

    docstrings.keep_params('ArrayList.rename.parameters', 'new_name')

    @docstrings.dedent
    def append(self, value, new_name=False):
        """
        Append a new array to the list

        Parameters
        ----------
        value: InteractiveBase
            The data object to append to this list
        %(ArrayList.rename.parameters.new_name)s

        Raises
        ------
        %(ArrayList.rename.raises)s

        See Also
        --------
        list.append, extend, rename"""
        arr, renamed = self.rename(value, new_name)
        if renamed is not None:
            super(ArrayList, self).append(value)

    @docstrings.dedent
    def extend(self, iterable, new_name=False, force=False):
        """
        Add further arrays from an iterable to this list

        Parameters
        ----------
        iterable
            Any iterable that contains :class:`InteractiveBase` instances
        %(ArrayList.rename.parameters.new_name)s
        force: bool
            If True, all objects in `iterable` are extended if they are
            already in the list or not

        Raises
        ------
        %(ArrayList.rename.raises)s

        See Also
        --------
        list.extend, append, rename"""
        # extend those arrays that aren't alredy in the list
        if not force:
            super(ArrayList, self).extend(t[0] for t in filter(
                lambda t: t[1] is not None,
                (self.rename(arr, new_name) for arr in iterable)))
        else:
            super(ArrayList, self).extend(t[0] for t in (
                self.rename(arr, new_name) for arr in iterable))

    def remove(self, arr):
        """Removes an array from the list

        Parameters
        ----------
        arr: str or :class:`InteractiveBase`
            The array name or the data object in this list to remove

        Raises
        ------
        ValueError
            If no array with the specified array name is in the list"""
        name = arr if isinstance(arr, six.string_types) else arr.arr_name
        for i, arr in enumerate(self):
            if arr.arr_name == name:
                del self[i]
                return
        raise ValueError(
            "Not array found with name {0}".format(name))


class InteractiveList(ArrayList, InteractiveBase):
    """List of :class:`InteractiveArray` instances that can be plotted itself

    This class combines the :class:`ArrayList` and the interactive plotting
    through :class:`psyplot.plotter.Plotter` classes. It is mainly used by the
    :mod:`psyplot.plotter.simple` module"""

    @property
    def auto_update(self):
        """:class:`bool`. Boolean controlling whether the :meth:`start_update`
        method is automatically called by the :meth:`update` method"""
        return self._auto_update

    @auto_update.setter
    def auto_update(self, value):
        ArrayList.auto_update.fset(self, value)
        InteractiveBase.auto_update.fset(self, value)

    @property
    @docstrings
    def _njobs(self):
        """%(InteractiveBase._njobs)s"""
        ret = super(self.__class__, self)._njobs or [0]
        ret[0] += 1
        return ret

    docstrings.delete_params('InteractiveBase.parameters', 'auto_update')

    @docstrings.dedent
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        %(ArrayList.parameters)s
        %(InteractiveBase.parameters.no_auto_update)s"""
        ibase_kwargs, array_kwargs = sort_kwargs(
            kwargs, ['plotter', 'arr_name'])
        self._registered_updates = {}
        InteractiveBase.__init__(self, **ibase_kwargs)
        ArrayList.__init__(self, *args, **kwargs)

    @docstrings.dedent
    def _register_update(self, method='isel', replot=False, dims={}, fmt={},
                         force=False, todefault=False):
        """
        Register new dimensions and formatoptions for updating

        Parameters
        ----------
        %(InteractiveArray._register_update.parameters)s"""
        ArrayList._register_update(self, method=method, dims=dims)
        InteractiveBase._register_update(self, fmt=fmt, todefault=todefault,
                                         replot=bool(dims) or replot,
                                         force=force)

    @docstrings.dedent
    def start_update(self, draw=True, queues=None):
        """
        Conduct the formerly registered updates

        This method conducts the updates that have been registered via the
        :meth:`update` method. You can call this method if the
        :attr:`auto_update` attribute of this instance and the `auto_update`
        parameter in the :meth:`update` method has been set to False

        Parameters
        ----------
        %(InteractiveBase.start_update.parameters)s

        Returns
        -------
        %(InteractiveBase.start_update.returns)s

        See Also
        --------
        :attr:`auto_update`, update
        """
        if queues is not None:
            queues[0].get()
        for arr in self:
            arr.start_update(draw=False)
        if queues is not None:
            queues[0].task_done()
        return InteractiveBase.start_update(self, draw=draw)

    def to_dataframe(self):
        def to_df(arr):
            df = arr.to_pandas()
            if hasattr(df, 'to_frame'):
                df = df.to_frame()
            return df.rename(columns={df.keys()[0]: arr.arr_name})
        if len(self) == 1:
            return self[0].to_pandas().to_frame()
        else:
            df = to_df(self[0])
            for arr in self[1:]:
                df = df.merge(to_df(arr), left_index=True, right_index=True)
            return df

    docstrings.delete_params('ArrayList.from_dataset.parameters', 'plotter')
    docstrings.delete_kwargs('ArrayList.from_dataset.other_parameters',
                             None, 'kwargs')

    @classmethod
    @docstrings.dedent
    def from_dataset(cls, *args, **kwargs):
        """
        Create an InteractiveList instance from the given base dataset

        Parameters
        ----------
        %(ArrayList.from_dataset.parameters.no_plotter)s
        plotter: psyplot.plotter.Plotter
            The plotter instance that is used to visualize the data in this
            list

        Other Parameters
        ----------------
        %(ArrayList.from_dataset.other_parameters.no_args_kwargs)s
        ``**kwargs``
            Further keyword arguments may point to any of the dimensions of the
            data (see `dims`)

        Returns
        -------
        %(ArrayList.from_dataset.returns)s"""
        plotter = kwargs.pop('plotter', None)
        plot = kwargs.pop('plot', True)
        instance = super(InteractiveList, cls).from_dataset(*args, **kwargs)
        if plotter is not None:
            plotter.initialize_plot(instance, plot=plot)
        return instance


class _MissingModule(object):
    """Class that can be used if an optional module is not avaible.

    This class raises an error if any attribute is accessed or it is called"""
    def __init__(self, error):
        """
        Parameters
        ----------
        error: ImportError
            The error that has been raised when tried to import the module"""
        self.error = error

    def __getattr__(self, attr):
        raise self.error.__class__(self.error.message)

    def __call__(self, *args, **kwargs):
        raise self.error.__class__(self.error.message)


class CFDecoder(object):
    """
    Class that interpretes the coordinates and attributes accordings to
    cf-conventions"""
    def __init__(self, ds=None, x=None, y=None, z=None, t=None):
        self.ds = ds
        self.x = rcParams['decoder.x'] if x is None else set(x)
        self.y = rcParams['decoder.y'] if x is None else set(y)
        self.z = rcParams['decoder.z'] if x is None else set(z)
        self.t = rcParams['decoder.t'] if x is None else set(t)

    @staticmethod
    @docstrings.get_sectionsf('CFDecoder.decode_coords')
    def decode_coords(ds, gridfile=None, inplace=True):
        """
        Sets the coordinates and bounds in a dataset

        This static method sets those coordinates and bounds that are marked
        marked in the netCDF attributes as coordinates in :attr:`ds` (without
        deleting them from the variable attributes because this information is
        necessary for visualizing the data correctly)

        Parameters
        ----------
        ds: xray.Dataset
            The dataset to decode
        gridfile: str
            The path to a separate grid file or a xray.Dataset instance which
            may store the coordinates used in `ds`
        inplace: bool, optional
            If True, `ds` is modified in place

        Returns
        -------
        xray.Dataset
            `ds` with additional coordinates"""
        def add_attrs(obj):
            if 'coordinates' in obj.attrs:
                extra_coords.update(obj.attrs['coordinates'].split())
            if 'bounds' in obj.attrs:
                extra_coords.add(obj.attrs['bounds'])
        if gridfile is not None and not isinstance(gridfile, xray.Dataset):
            gridfile = xray.open_dataset(gridfile)
        extra_coords = set(ds.coords)
        for k, v in six.iteritems(ds.variables):
            add_attrs(v)
        add_attrs(ds)
        if gridfile is not None:
            ds = ds.update({k: v for k, v in six.iteritems(gridfile.variables)
                            if k in extra_coords}, inplace=inplace)
        ds = ds.set_coords(extra_coords.intersection(ds.variables),
                           inplace=inplace)
        return ds

    @docstrings.get_sectionsf('CFDecoder.is_triangular', sections=[
        'Parameters', 'Returns'])
    @dedent
    def is_triangular(self, var):
        """
        Test if a variable is on a triangular grid

        This method first checks the `grid_type` attribute of the variable (if
        existent) whether it is equal to ``"unstructered"``, then it checks
        whether the bounds are not two-dimensional.

        Parameters
        ----------
        var: xray.Variable or xray.DataArray
            The variable to check

        Returns
        -------
        bool
            True, if the grid is triangular, else False"""
        return str(var.attrs.get('grid_type')) == 'unstructured' or \
            self._check_triangular_bounds(var)[0]

    @docstrings.get_sectionsf('CFDecoder._check_triangular_bounds', sections=[
        'Parameters', 'Returns'])
    @dedent
    def _check_triangular_bounds(self, var, axis='x'):
        """
        Checks whether the bounds in the variable attribute are triangular

        Parameters
        ----------
        var: xray.Variable or xray.DataArray
            The variable to check
        axis: {'x', 'y'}
            The spatial axis to check

        Returns
        -------
        bool or None
            True, if unstructered, None if it could not be determined
        xray.Coordinate or None
            the bounds corrdinate (if existent)"""
        coord = self.get_variable_by_axis(var, axis)
        if coord is not None:
            bounds = coord.attrs.get('bounds')
            if bounds is not None:
                bounds = self.ds.coords.get(bounds)
                if bounds is not None:
                    return bounds.shape[-1] > 2, bounds
                else:
                    return None, bounds
        return None, None

    @docstrings.dedent
    def is_unstructured(self, *args, **kwargs):
        """
        Test if a variable is on an unstructered grid

        Parameters
        ----------
        %(CFDecoder.is_triangular.parameters)s

        Returns
        -------
        %(CFDecoder.is_triangular.returns)s

        Notes
        -----
        Currently this is the same as :meth:`is_triangular` method, but may
        change in the future to support hexagonal grids"""
        return self.is_triangular(*args, **kwargs)

    @docstrings.dedent
    def _check_unstructered_bounds(self, *args, **kwargs):
        """
        Checks whether the bounds in the variable attribute are triangular

        Parameters
        ----------
        %(CFDecoder._check_triangular_bounds.parameters)s

        Returns
        -------
        %(CFDecoder._check_triangular_bounds.returns)s

        Notes
        -----
        Currently this is the same as :meth:`_check_triangular_bounds` method,
        but may change in the future to support hexagonal grids"""
        return self._check_triangular_bounds(*args, **kwargs)

    def get_variable_by_axis(self, var, axis, coords=None):
        """Return the coordinate matching the specified axis

        This method uses to ``'axis'`` attribute in coordinates to return the
        corresponding coordinate of the given variable

        Possible types
        --------------
        var: xray.Variable
            The variable to get the dimension for
        axis: {'x', 'y', 'z', 't'}
            The axis string that identifies the dimension
        coords: dict
            Coordinates to use. If None, the coordinates of the dataset in the
            :attr:`ds` attribute are used.

        Returns
        -------
        xray.Coordinate or None
            The coordinate for `var` that matches the given `axis` or None if
            no coordinate with the right `axis` could be found.

        See Also
        --------
        get_x, get_y, get_z, get_t"""

        if not isinstance(axis, six.string_types) or not axis:
            raise ValueError("Axis must be one of X, Y, Z, T, not {0}".format(
                axis))
        axis = axis.lower()
        # we first check for the dimensions and then for the coordinates
        # attribute
        coords = coords or self.ds.coords
        coord_names = var.attrs.get('coordinates', '').split()
        if not coord_names:
            return
        for coord in map(lambda dim: coords[dim], filter(
                lambda dim: dim in coords, chain(
                    var.dims, coord_names))):
            if coord.attrs.get('axis', '').lower() == axis:
                return coord
        #: If the coordinates is specified but the coordinate
        #: variables themselves have no 'axis' attribute, we interpret the
        #: coordinates such that x: -1, y: -2, z: -3, t: -4
        if axis == 'x':
            return coords.get(coord_names[-2 if len(coord_names) >= 2 else -1])
        elif axis == 'y':
            return coords.get(coord_names[-1])
        elif axis == 'z':
            if len(coord_names) > 2:
                i = -3
            elif len(coord_names) > 1:
                i = -2
            else:
                i = -1
            return coords.get(coord_names[i])
        elif axis == 't':
            return coords.get(coord_names[0])

    def get_x(self, var, coords=None):
        """Get the x-coordinate of a variable

        This method searches for the x-coordinate in the :attr:`ds`. It first
        checks whether there is one dimension that holds an ``'axis'``
        attribute with 'X', otherwise it looks whether there is an intersection
        between the :attr:`x` attribute and the variables dimensions, otherwise
        it returns the coordinate corresponding to the last dimension of `var`

        Possible types
        --------------
        var: xray.Variable
            The variable to get the x-coordinate for
        coords: dict
            Coordinates to use. If None, the coordinates of the dataset in the
            :attr:`ds` attribute are used.

        Returns
        -------
        xray.Coordinate
            The x-coordinate"""
        coords = coords or self.ds.coords
        coord = self.get_variable_by_axis(var, 'x', coords)
        if coord is not None:
            return coord
        return coords.get(self.get_xname(var))

    def get_xname(self, var, coords=None):
        """Get the name of the x-dimension

        This method gives the name of the x-dimension (which is not necessarily
        the name of the coordinate if the variable has a coordinate attribute)

        Parameters
        ----------
        var: xray.Variables
            The variable to get the dimension for
        coords: dict
            The coordinates to use for checking the axis attribute. If None,
            they are not used

        Returns
        -------
        str
            The coordinate name

        See Also
        --------
        get_x"""
        if coords is not None:
            coord = self.get_variable_by_axis(var, 'x', coords)
            if coord is not None and coord.name in var.dims:
                return coord.name
        dimlist = list(self.x.intersection(var.dims))
        if dimlist:
            if len(dimlist) > 1:
                warn("Found multiple matches for x coordinate in the variable:"
                     "%s. I use %s" % (', '.join(dimlist), dimlist[0]),
                     PsyPlotRuntimeWarning)
            return dimlist[0]
        # otherwise we return the coordinate in the last position
        return var.dims[-1]

    def get_y(self, var, coords=None):
        """Get the y-coordinate of a variable

        This method searches for the y-coordinate in the :attr:`ds`. It first
        checks whether there is one dimension that holds an ``'axis'``
        attribute with 'Y', otherwise it looks whether there is an intersection
        between the :attr:`y` attribute and the variables dimensions, otherwise
        it returns the coordinate corresponding to the second last dimension of
        `var` (or the last if the dimension of var is one-dimensional)

        Possible types
        --------------
        var: xray.Variable
            The variable to get the y-coordinate for
        coords: dict
            Coordinates to use. If None, the coordinates of the dataset in the
            :attr:`ds` attribute are used.

        Returns
        -------
        xray.Coordinate
            The y-coordinate"""
        coords = coords or self.ds.coords
        coord = self.get_variable_by_axis(var, 'y', coords)
        if coord is not None:
            return coord
        return coords.get(self.get_yname(var))

    def get_yname(self, var, coords=None):
        """Get the name of the y-dimension

        This method gives the name of the y-dimension (which is not necessarily
        the name of the coordinate if the variable has a coordinate attribute)

        Parameters
        ----------
        var: xray.Variables
            The variable to get the dimension for
        coords: dict
            The coordinates to use for checking the axis attribute. If None,
            they are not used

        Returns
        -------
        str
            The coordinate name

        See Also
        --------
        get_y"""
        if coords is not None:
            coord = self.get_variable_by_axis(var, 'y', coords)
            if coord is not None and coord.name in var.dims:
                return coord.name
        dimlist = list(self.y.intersection(var.dims))
        if dimlist:
            if len(dimlist) > 1:
                warn("Found multiple matches for y coordinate in the variable:"
                     "%s. I use %s" % (', '.join(dimlist), dimlist[0]),
                     PsyPlotRuntimeWarning)
            return dimlist[0]
        # otherwise we return the coordinate in the last or second last
        # position
        if self.is_unstructured(var):
            return var.dims[-1]
        return var.dims[-2 if var.ndim > 1 else -1]

    def get_z(self, var, coords=None):
        """Get the vertical (z-) coordinate of a variable

        This method searches for the z-coordinate in the :attr:`ds`. It first
        checks whether there is one dimension that holds an ``'axis'``
        attribute with 'Z', otherwise it looks whether there is an intersection
        between the :attr:`z` attribute and the variables dimensions, otherwise
        it returns the coordinate corresponding to the third last dimension of
        `var` (or the second last or last if var is two or one-dimensional)

        Possible types
        --------------
        var: xray.Variable
            The variable to get the z-coordinate for
        coords: dict
            Coordinates to use. If None, the coordinates of the dataset in the
            :attr:`ds` attribute are used.

        Returns
        -------
        xray.Coordinate
            The z-coordinate"""
        coords = coords or self.ds.coords
        coord = self.get_variable_by_axis(var, 'z', coords)
        if coord is not None:
            return coord
        return coords.get(self.get_zname(var))

    def get_zname(self, var, coords=None):
        """Get the name of the z-dimension

        This method gives the name of the z-dimension (which is not necessarily
        the name of the coordinate if the variable has a coordinate attribute)

        Parameters
        ----------
        var: xray.Variables
            The variable to get the dimension for
        coords: dict
            The coordinates to use for checking the axis attribute. If None,
            they are not used

        Returns
        -------
        str
            The coordinate name

        See Also
        --------
        get_z"""
        if coords is not None:
            coord = self.get_variable_by_axis(var, 'z', coords)
            if coord is not None and coord.name in var.dims:
                return coord.name
        dimlist = list(self.z.intersection(var.dims))
        if dimlist:
            if len(dimlist) > 1:
                warn("Found multiple matches for z coordinate in the variable:"
                     "%s. I use %s" % (', '.join(dimlist), dimlist[0]),
                     PsyPlotRuntimeWarning)
            return dimlist[0]
        # otherwise we return the coordinate in the third last position
        is_unstructured = self.is_unstructured(var)
        if var.ndim > 2:
            i = -3 if not is_unstructured else -2
        elif var.ndim > 1:
            i = -2 if not is_unstructured else -1
        else:
            i = -1
        return var.dims[i]

    def get_t(self, var, coords=None):
        """Get the time coordinate of a variable

        This method searches for the time coordinate in the :attr:`ds`. It
        first checks whether there is one dimension that holds an ``'axis'``
        attribute with 'T', otherwise it looks whether there is an intersection
        between the :attr:`t` attribute and the variables dimensions, otherwise
        it returns the coordinate corresponding to the first dimension of `var`

        Possible types
        --------------
        var: xray.Variable
            The variable to get the time coordinate for
        coords: dict
            Coordinates to use. If None, the coordinates of the dataset in the
            :attr:`ds` attribute are used.

        Returns
        -------
        xray.Coordinate
            The time coordinate"""
        coords = coords or self.ds.coords
        coord = self.get_variable_by_axis(var, 't', coords)
        if coord is not None:
            return coord
        dimlist = list(self.t.intersection(var.dims))
        if dimlist:
            if len(dimlist) > 1:
                warn("Found multiple matches for time coordinate in the "
                     "variable: %s. I use %s" % (
                         ', '.join(dimlist), dimlist[0]),
                     PsyPlotRuntimeWarning)
            return coords[dimlist[0]]
        # otherwise we return the coordinate in the first position
        return coords.get(self.get_tname(var))

    def get_tname(self, var, coords=None):
        """Get the name of the t-dimension

        This method gives the name of the time dimension (which is not
        necessarily the name of the coordinate if the variable has a coordinate
        attribute)

        Parameters
        ----------
        var: xray.Variables
            The variable to get the dimension for
        coords: dict
            The coordinates to use for checking the axis attribute. If None,
            they are not used

        Returns
        -------
        str
            The coordinate name

        See Also
        --------
        get_t"""
        if coords is not None:
            coord = self.get_variable_by_axis(var, 't', coords)
            if coord is not None and coord.name in var.dims:
                return coord.name
        dimlist = list(self.t.intersection(var.dims))
        if dimlist:
            if len(dimlist) > 1:
                warn("Found multiple matches for t coordinate in the variable:"
                     "%s. I use %s" % (', '.join(dimlist), dimlist[0]),
                     PsyPlotRuntimeWarning)
            return dimlist[0]
        # otherwise we return the coordinate in the first position
        return var.dims[0]

    def get_idims(self, arr, coords=None):
        """Get the coordinates in the :attr:`ds` dataset as int or slice

        This method returns a mapping from the coordinate names of the given
        `arr` to an integer, slice or an array of integer that represent the
        coordinates in the :attr:`ds` dataset and can be used to extract the
        given `arr` via the :meth:`xray.Dataset.isel` method.

        Parameters
        ----------
        arr: xray.DataArray
            The data array for which to get the dimensions as integers, slices
            or list of integers from the dataset in the :attr:`base` attribute

        Returns
        -------
        dict
            Mapping from coordinate name to integer, list of integer or slice

        See Also
        --------
        xray.Dataset.isel, InteractiveArray.idims"""
        if coords is None:
            coord_items = six.iteritems(arr.coords)
        else:
            coord_items = ((label, coord) for label, coord in six.iteritems(
                arr.coords) if label in coords)
        return dict(
                (label, get_index_from_coord(coord, self.ds.indexes[label]))
                for label, coord in coord_items if label in self.ds.indexes)

    @docstrings.get_sectionsf('CFDecoder.get_plotbounds', sections=[
        'Parameters', 'Returns'])
    @dedent
    def get_plotbounds(self, coord, kind=None, ignore_shape=False):
        """
        Get the bounds of a coordinate

        This method first checks the ``'bounds'`` attribute of the given
        `coord` and if it fails, it calculates them.

        Parameters
        ----------
        coord: xray.Coordinate
            The coordinate to get the bounds for
        kind: str
            The interpolation method (see :func:`scipy.interpolate.interp1d`)
            that is used in case of a 2-dimensional coordinate
        ignore_shape: bool
            If True and the `coord` has a ``'bounds'`` attribute, this
            attribute is returned without further check. Otherwise it is tried
            to bring the ``'bounds'`` into a format suitable for (e.g.) the
            :func:`matplotlib.pyplot.pcolormesh` function.

        Returns
        -------
        bounds: np.ndarray
            The bounds with the same number of dimensions as `coord` but one
            additional array (i.e. if `coord` has shape (4, ), `bounds` will
            have shape (5, ) and if `coord` has shape (4, 5), `bounds` will
            have shape (5, 6)"""
        if 'bounds' in coord.attrs:
            bounds = self.ds.coords[coord.attrs['bounds']]
            if ignore_shape:
                return bounds.values.ravel()
            if not bounds.shape[:-1] == coord.shape:
                bounds = self.ds.isel(*self.get_idims(coord))
            try:
                return self._get_plotbounds_from_cf(coord, bounds)
            except ValueError as e:
                warn(e.message + " Bounds are calculated automatically!")
        return self._infer_interval_breaks(coord, kind=kind)

    @staticmethod
    @docstrings.dedent
    def _get_plotbounds_from_cf(coord, bounds):
        """
        Get plot bounds from the bounds stored as defined by CFConventions

        Parameters
        ----------
        coord: xray.Coordinate
            The coordinate to get the bounds for
        bounds: xray.DataArray
            The bounds as inferred from the attributes of the given `coord`

        Returns
        -------
        %(CFDecoder.get_plotbounds.returns)s

        Notes
        -----
        this currently only works for rectilinear grids"""
        if bounds.shape[:-1] != coord.shape or bounds.shape[-1] != 2:
            raise ValueError(
                "Cannot interprete bounds with shape {0} for {1} "
                "coordinate with shape {2}.".format(
                    bounds.shape, coord.name, coord.shape))
        ret = zeros(tuple(map(lambda i: i+1, coord.shape)))
        ret[tuple(map(slice, coord.shape))] = bounds[..., 0]
        last_slices = tuple(slice(-1, None) for _ in coord.shape)
        ret[last_slices] = bounds[tuple(chain(last_slices, [1]))]
        return ret

    def get_triangles(self, var, coords=None, convert_radian=True,
                      copy=False, src_crs=None, target_crs=None):
        """Get the triangles for the variable

        Parameters
        ----------
        var: xray.Variable or xray.DataArray
            The variable to use
        coords: dict
            Alternative coordinates to use. If None, the coordinates of the
            :attr:`ds` dataset are used
        convert_radian: bool
            If True and the coordinate has units in 'radian', those are
            converted to degrees
        copy: bool
            If True, vertice arrays are copied
        src_crs: cartopy.crs.Crs
            The source projection of the data. If not None, a transformation
            to the given `target_crs` will be done
        target_crs: cartopy.crs.Crs
            The target projection for which the triangles shall be transformed.
            Must only be provided if the `src_crs` is not None.

        Returns
        -------
        matplotlib.tri.Triangulation
            The spatial triangles of the variable

        Raises
        ------
        ValueError
            If `src_crs` is not None and `target_crs` is None"""
        from matplotlib.tri import Triangulation

        def get_vertices(axis):
            bounds = self._check_triangular_bounds(var, axis)[1]
            if coords is not None:
                bounds = coords.get(bounds.name, bounds)
            vertices = bounds.values.ravel()
            if convert_radian:
                coord = getattr(self, 'get_' + axis)(var)
                if coord.attrs.get('units') == 'radian':
                    vertices = vertices * 180. / pi
            return vertices if not copy else vertices.copy()

        xvert = get_vertices('x')
        yvert = get_vertices('y')
        if src_crs is not None and src_crs != target_crs:
            if target_crs is None:
                raise ValueError(
                    "Found %s for the source crs but got None for the "
                    "target_crs!" % (src_crs, ))
            arr = target_crs.transform_points(src_crs, xvert, yvert)
            xvert = arr[:, 0]
            yvert = arr[:, 1]
        triangles = reshape(range(len(xvert)), (len(xvert) / 3, 3))
        return Triangulation(xvert, yvert, triangles)

    docstrings.delete_params(
        'CFDecoder.get_plotbounds.parameters', 'ignore_shape')

    @staticmethod
    def _infer_interval_breaks(coord, kind=None):
        """
        Interpolate the bounds from the data in coord

        Parameters
        ----------
        %(CFDecoder.get_plotbounds.parameters.no_ignore_shape)s

        Returns
        -------
        %(CFDecoder.get_plotbounds.returns)s

        Notes
        -----
        this currently only works for rectilinear grids"""
        if coord.ndim == 1:
            return _infer_interval_breaks(coord)
        elif coord.ndim == 2:
            from scipy.interpolate import interp2d
            kind = kind or rcParams['decoder.interp_kind']
            x, y = map(arange, coord.shape)
            new_x, new_y = map(_infer_interval_breaks, [x, y])
            return interp2d(x, y, asarray(coord), kind=kind)(new_x, new_y)

    @staticmethod
    @docstrings.dedent
    def decode_ds(ds, gridfile=None, inplace=False, decode_coords=True,
                  decode_times=True):
        """
        Static method to decode coordinates and time informations

        This method interpretes absolute time informations (stored with units
        ``'day as %Y%m%d.%f'``) and coordinates

        Parameters
        ----------
        %(CFDecoder.decode_coords.parameters)s
        decode_times : bool, optional
            If True, decode times encoded in the standard NetCDF datetime
            format into datetime objects. Otherwise, leave them encoded as
            numbers.
        decode_coords : bool, optional
            If True, decode the 'coordinates' attribute to identify coordinates
            in the resulting dataset."""
        if decode_coords:
            ds = CFDecoder.decode_coords(ds, gridfile=gridfile,
                                         inplace=inplace)
        if decode_times:
            for k, v in six.iteritems(ds.variables):
                if v.attrs.get('units', '') == 'day as %Y%m%d.%f':
                    ds = ds.update({k: AbsoluteTimeDecoder(v)},
                                   inplace=inplace)
        return ds

    def correct_dims(self, var, dims={}):
        """Expands the dimensions to match the dims in the variable

        Parameters
        ----------
        var: xray.Variable
            The variable to get the data for
        dims: dict
            a mapping from dimension to the slices"""
        method_mapping = {'x': self.get_xname,
                          'z': self.get_zname, 't': self.get_tname}
        if self.is_unstructured(var):  # we assume a one-dimensional grid
            method_mapping['y'] = self.get_xname
        else:
            method_mapping['y'] = self.get_yname
        for key in six.iterkeys(dims.copy()):
            if key in method_mapping and key not in var.dims:
                dim_name = method_mapping[key](var, self.ds.coords)
                if dim_name in dims:
                    dims.pop(key)
                else:
                    dims[method_mapping[key](var)] = dims.pop(key)
        return dims


#: mapping that translates datetime format strings to regex patterns
t_patterns = {
        '%Y': '[0-9]{4}',
        '%m': '[0-9]{1,2}',
        '%d': '[0-9]{1,2}',
        '%H': '[0-9]{1,2}',
        '%M': '[0-9]{1,2}',
        '%S': '[0-9]{1,2}',
    }


@docstrings.get_sectionsf('get_tdata')
@dedent
def get_tdata(t_format, files):
    """
    Get the time information from file names

    Parameters
    ----------
    t_format: str
        The string that can be used to get the time information in the files.
        Any numeric datetime format string (e.g. %Y, %m, %H) can be used, but
        not non-numeric strings like %b, etc. See [1]_ for the datetime format
        strings
    files: list of str
        The that contain the time informations

    Returns
    -------
    pandas.Index
        The time coordinate
    list of str
        The file names as they are sorten in the returned index

    References
    ----------
    .. [1] https://docs.python.org/2/library/datetime.html"""
    def median(arr):
        return arr.min() + (arr.max() - arr.min())/2
    import re
    from numpy import datetime64, array, argsort
    import datetime as dt
    from pandas import Index
    t_pattern = t_format
    for fmt, patt in t_patterns.items():
        t_pattern = t_pattern.replace(fmt, patt)
    t_pattern = re.compile(t_pattern)
    time = range(len(files))
    for i, f in enumerate(files):
        time[i] = median(array(list(map(
            lambda s: datetime64(dt.datetime.strptime(s, t_format)),
            t_pattern.findall(f)))))
    ind = argsort(time)  # sort according to time
    files = array(files)[ind]
    time = array(time)[ind]
    return Index(time, name='time'), files


def decode_absolute_time(times):
    def decode(t):
        day, sub = re.findall('(\d+)(\.\d+)', t)[0]
        return datetime64(
            datetime.strptime(day, "%Y%m%d") + timedelta(days=float(sub)))
    times = asarray(times, dtype=str)
    return vectorize(decode, [datetime64])(times)


class AbsoluteTimeDecoder(NDArrayMixin):

    def __init__(self, array):
        self.array = array
        example_value = first_n_items(array, 1) or 0
        try:
            result = decode_absolute_time(example_value)
        except Exception:
            logger.error("Could not interprete absolute time values!")
            raise
        else:
            self._dtype = getattr(result, 'dtype', dtype('object'))

    @property
    def dtype(self):
        return self._dtype

    def __getitem__(self, key):
        return decode_absolute_time(self.array[key])

docstrings.keep_params('CFDecoder.decode_coords.parameters', 'gridfile')
docstrings.get_sections(dedents(xray.open_dataset.__doc__[
    xray.open_dataset.__doc__.find('\n') + 1:]), 'xray.open_dataset')
docstrings.delete_params('xray.open_dataset.parameters', 'engine')


@docstrings.get_sectionsf('open_dataset')
@docstrings.dedent
def open_dataset(filename_or_obj, decode_cf=True, decode_times=True,
                 decode_coords=True, engine=None, gridfile=None, **kwargs):
    """
    Open an instance of :class:`xray.Dataset`.

    This method has the same functionality as the :func:`xray.open_dataset`
    method except that is supports an additional 'gdal' engine to open
    gdal Rasters (e.g. GeoTiffs) and that is supports absolute time units like
    ``'day as %Y%m%d.%f'`` (if `decode_cf` and `decode_times` are True).

    Parameters
    ----------
    %(xray.open_dataset.parameters.no_engine)s
    engine: {'netcdf4', 'scipy', 'pydap', 'h5netcdf', 'gdal'}, optional
        Engine to use when reading netCDF files. If not provided, the default
        engine is chosen based on available dependencies, with a preference for
        'netcdf4'.
    %(CFDecoder.decode_coords.parameters.gridfile)s

    Returns
    -------
    xray.Dataset
        The dataset that contains the variables from `filename_or_obj`"""
    if engine == 'gdal':
        from .gdal_store import GdalStore
        filename_or_obj = GdalStore(filename_or_obj)
        engine = None
    ds = xray.open_dataset(filename_or_obj, decode_cf=decode_cf,
                           decode_coords=False, engine=engine, **kwargs)
    if decode_cf:
        ds = CFDecoder.decode_ds(
            ds, decode_coords=decode_coords, decode_times=decode_times,
            gridfile=gridfile, inplace=True)
    return ds


docstrings.get_sections(dedents(xray.open_mfdataset.__doc__[
    xray.open_mfdataset.__doc__.find('\n') + 1:]), 'xray.open_mfdataset')
docstrings.delete_params('xray.open_mfdataset.parameters', 'engine')
docstrings.keep_params('get_tdata.parameters', 't_format')

docstrings.params['xray.open_mfdataset.parameters.no_engine'] = \
    docstrings.params['xray.open_mfdataset.parameters.no_engine'].replace(
        '**kwargs', '``**kwargs``').replace('"path/to/my/files/*.nc"',
                                            '``"path/to/my/files/*.nc"``')


docstrings.keep_params('open_dataset.parameters', 'engine')


@docstrings.dedent
def open_mfdataset(paths, decode_cf=True, decode_times=True,
                   decode_coords=True, engine=None, gridfile=None,
                   t_format=None, **kwargs):
    """
    Open multiple files as a single dataset.

    This function is essentially the same as the :func:`xray.open_mfdataset`
    function but (as the :func:`open_dataset`) supports additional decoding
    and the ``'gdal'`` engine.
    You can further specify the `t_format` parameter to get the time
    information from the files and use the results to concatenate the files

    Parameters
    ----------
    %(xray.open_mfdataset.parameters.no_engine)s
    %(open_dataset.parameters.engine)s
    %(get_tdata.parameters.t_format)s
    %(CFDecoder.decode_coords.parameters.gridfile)s

    Returns
    -------
    xray.Dataset
        The dataset that contains the variables from `filename_or_obj`"""
    if t_format is not None or engine == 'gdal':
        if isinstance(paths, six.string_types):
            paths = sorted(glob(paths))
        if not paths:
            raise IOError('no files to open')
    if t_format is not None:
        time, paths = get_tdata(t_format, paths)
        kwargs['concat_dim'] = time
    if engine == 'gdal':
        from .gdal_store import GdalStore
        paths = list(map(GdalStore, paths))
        engine = None
        kwargs['lock'] = False

    ds = xray.open_mfdataset(
        paths, decode_cf=decode_cf, decode_times=decode_times, engine=engine,
        decode_coords=False, **kwargs)
    if decode_cf:
        return CFDecoder.decode_ds(ds, gridfile=gridfile, inplace=True,
                                   decode_coords=decode_coords,
                                   decode_times=decode_times)
    return ds
