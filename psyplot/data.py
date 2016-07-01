import os
from threading import Thread
from functools import partial
from glob import glob
from importlib import import_module
import re
import six
from copy import deepcopy
from collections import defaultdict
from itertools import chain, product, repeat, starmap, count, cycle
from difflib import get_close_matches
import xarray
from xarray.core.utils import NDArrayMixin
from xarray.core.formatting import first_n_items, format_item
import xarray.backends.api as xarray_api
from pandas import to_datetime
import numpy as np
from datetime import datetime, timedelta
import logging
from psyplot.config.rcsetup import rcParams, safe_list
from psyplot.docstring import dedent, docstrings, dedents
from psyplot.compat.pycompat import (
    zip, map, isstring, OrderedDict, filter, range, getcwd, filterfalse,
    Queue)
from psyplot.warning import warn, PsyPlotRuntimeWarning


# No data variable. This is used for filtering if an attribute could not have
# been accessed
_NODATA = object


logger = logging.getLogger(__name__)


_ds_counter = count(1)


class _TempBool(object):
    """Wrapper around a boolean defining an __enter__ and __exit__ method

    Notes
    -----
    If you want to use this class as an instance property, rather use the
    :func:`_temp_bool_prop` because this class as a descriptor is ment to be a
    class descriptor"""

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

    def __repr__(self):
        return repr(bool(self))

    def __str__(self):
        return str(bool(self))

    def __call__(self, value=None):
        """
        Parameters
        ----------
        value: bool or None
            If None, the current value will be negated. Otherwise the current
            value of this instance is set to the given `value`"""
        if value is None:
            self.value = not self.value
        else:
            self.value = value

    def __get__(self, instance, owner):
        return self

    def __set__(self, instance, value):
        self.value = value


def _temp_bool_prop(propname, doc="", default=False):
    """Creates a property that uses the :class:`_TempBool` class

    Parameters
    ----------
    propname: str
        The attribute name to use. The _TempBool instance will be stored in the
        ``'_' + propname`` attribute of the corresponding instance
    doc: str
        The documentation of the property
    default: bool
        The default value of the _TempBool class"""
    def getx(self):
        if getattr(self, '_' + propname, None) is not None:
            return getattr(self, '_' + propname)
        else:
            setattr(self, '_' + propname, _TempBool(default))
        return getattr(self, '_' + propname)

    def setx(self, value):
        getattr(self, propname).value = bool(value)

    def delx(self):
        getattr(self, propname).value = default

    return property(getx, setx, delx, doc)


def _no_auto_update_getter(self):
    """:class:`bool`. Boolean controlling whether the :meth:`start_update`
    method is automatically called by the :meth:`update` method


    Examples
    --------
    You can disable the automatic update via

        >>> with data.no_auto_update:
        ...     data.update(time=1)
        ...     data.start_update()

    To permanently disable the automatic update, simply set

        >>> data.no_auto_update = True
        >>> data.update(time=1)
        >>> data.no_auto_update = False  # reenable automatical update"""
    if getattr(self, '_no_auto_update', None) is not None:
        return self._no_auto_update
    else:
        self._no_auto_update = _TempBool()
    return self._no_auto_update


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


def is_remote_url(path):
    return bool(re.search('^https?\://', path))


@docstrings.get_sectionsf('check_key', sections=['Parameters', 'Returns',
                                                 'Raises'])
@dedent
def check_key(key, possible_keys, raise_error=True,
              name='formatoption keyword',
              msg=("See show_fmtkeys function for possible formatopion "
                   "keywords"),
              *args, **kwargs):
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
    msg: str
        The additional message that shall be used if no close match to
        key is found
    ``*args,**kwargs``
        They are passed to the :func:`difflib.get_close_matches` function
        (i.e. `n` to increase the number of returned similar keys and
        `cutoff` to change the sensibility)

    Returns
    -------
    str
        The `key` if it is a valid string, else an empty string
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
            msg = ("Unknown %s %s! ") % (name, key) + msg
        if not raise_error:
            return '', similarkeys, msg
        raise KeyError(msg)
    else:
        return key, [key], ''


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

    Taken from xarray.plotting.plot module
    """
    coord = np.asarray(coord)
    deltas = 0.5 * (coord[1:] - coord[:-1])
    first = coord[0] - deltas[0]
    last = coord[-1] + deltas[-1]
    return np.r_[[first], coord[:-1] + deltas, [last]]


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
        dims.setdefault(key, val)
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
    step = np.unique(arr[1:] - arr[:-1])
    if len(step) == 1:
        return slice(arr[0], arr[-1] + 1, step[0])


def get_index_from_coord(coord, base_index):
    """Function to return the coordinate as integer, integer array or slice

    If `coord` is zero-dimensional, the corresponding integer in `base_index`
    will be supplied. Otherwise it is first tried to return a slice, if that
    does not work an integer array with the corresponding indices is returned.

    Parameters
    ----------
    coord: xarray.Coordinate or xarray.Variable
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
    values = np.array(list(map(lambda i: base_index.get_loc(i), values)))
    return is_slice(values) or values


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
    import datetime as dt
    from pandas import Index
    t_pattern = t_format
    for fmt, patt in t_patterns.items():
        t_pattern = t_pattern.replace(fmt, patt)
    t_pattern = re.compile(t_pattern)
    time = range(len(files))
    for i, f in enumerate(files):
        time[i] = median(np.array(list(map(
            lambda s: np.datetime64(dt.datetime.strptime(s, t_format)),
            t_pattern.findall(f)))))
    ind = np.argsort(time)  # sort according to time
    files = np.array(files)[ind]
    time = np.array(time)[ind]
    return Index(time, name='time'), files


docstrings.get_sections(
    dedents(xarray.Dataset.to_netcdf.__doc__.split('\n', 1)[1]),
    'xarray.Dataset.to_netcdf')


#: str. Name of the attribute that is assigned to a unique number for a dataset
PSYPLOT_ATTR = '_psyplot_num'


@docstrings.dedent
def to_netcdf(ds, *args, **kwargs):
    """
    Store the given dataset as a netCDF file

    This functions works essentially the same as the usual
    :meth:`xarray.Dataset.to_netcdf` method but can also encode absolute time
    units

    Parameters
    ----------
    ds: xarray.Dataset
        The dataset to store
    %(xarray.Dataset.to_netcdf.parameters)s
    """
    to_update = {}
    for v, obj in six.iteritems(ds.variables):
        units = obj.attrs.get('units', obj.encoding.get('units', None))
        if units == 'day as %Y%m%d.%f' and np.issubdtype(
                obj.dtype, np.datetime64):
            to_update[v] = xarray.Variable(
                obj.dims, AbsoluteTimeEncoder(obj), attrs=obj.attrs,
                encoding=obj.encoding)
    if to_update:
        ds = ds.update(to_update)
    if PSYPLOT_ATTR in ds.attrs:
        del ds.attrs[PSYPLOT_ATTR]
    return xarray_api.to_netcdf(ds, *args, **kwargs)


def _get_fname_netCDF4(store):
    """Try to get the file name from the NetCDF4DataStore store"""
    return getattr(store, '_filename', None)


def _get_fname_scipy(store):
    """Try to get the file name from the ScipyDataStore store"""
    try:
        return store.ds.filename
    except AttributeError:
        return None


def _get_fname_nio(store):
    """Try to get the file name from the NioDataStore store"""
    try:
        f = store.ds.file
    except AttributeError:
        return None
    s = str(f).splitlines()[0].rstrip().lstrip()
    if s.startswith('Nio file:'):
        try:
            return s.split('\t')[1]
        except IndexError:
            return None


class Signal(object):
    """Signal to connect functions to a specific event

    This class behaves almost similar to PyQt's
    :class:`PyQt4.QtCore.pyqtBoundSignal`
    """

    def __init__(self, name=None, cls_signal=False):
        self.name = name
        self.cls_signal = cls_signal
        self._connections = []

    def connect(self, func):
        if func not in self._connections:
            self._connections.append(func)

    def emit(self, *args, **kwargs):
        for func in self._connections[:]:
            func(*args, **kwargs)

    def disconnect(self, func=None):
        """Disconnect a function call to the signal. If None, all connections
        are disconnected"""
        if func is None:
            self._connections = []
        else:
            self._connections.remove(func)

    def __get__(self, instance, owner):
        if instance is None or self.cls_signal:
            return self
        ret = getattr(instance, self.name, None)
        if ret is None:
            setattr(instance, self.name, Signal(self.name))
            ret = getattr(instance, self.name, None)
        return ret


#: functions to use to extract the file name from a data store
get_fname_funcs = [_get_fname_netCDF4, _get_fname_scipy, _get_fname_nio]


@docstrings.get_sectionsf('get_filename_ds')
@docstrings.dedent
def get_filename_ds(ds, dump=True, paths=None, **kwargs):
    """
    Return the filename of the corresponding to a dataset

    This method returns the path to the `ds` or saves the dataset
    if there exists no filename

    Parameters
    ----------
    ds: xarray.Dataset
        The dataset you want the path information for
    dump: bool
        If True and the dataset has not been dumped so far, it is asked
        for a filename or the one generated by `paths` is used
    paths: iterable or True
        An iterator over filenames to use if a dataset has no filename.
        If paths is ``True``, an iterator over temporary files will be
        created without raising a warning

    Other Parameters
    ----------------
    ``**kwargs``
        Any other keyword for the :func:`to_netcdf` function
    %(xarray.Dataset.to_netcdf.parameters)s

    Returns
    -------
    str or None
        None, if the dataset has not yet been dumped to the harddisk and
        `dump` is False, otherwise the complete the path to the input
        file
    str
        The module of the :class:`xarray.backends.common.AbstractDataStore`
        instance that is used to hold the data
    str
        The class name of the
        :class:`xarray.backends.common.AbstractDataStore` instance that is
        used to open the data
    """
    from tempfile import NamedTemporaryFile

    def dump_nc():
        # make sure that the data store is not closed by providing a
        # write argument
        store = to_netcdf(ds, fname, writer=xarray_api.ArrayWriter(),
                          **kwargs)
        store_mod = store.__module__
        store_cls = store.__class__.__name__
        ds._file_obj = store
        return store_mod, store_cls

    def tmp_it():
        while True:
            yield NamedTemporaryFile(suffix='.nc').name

    fname = None
    if paths is True:
        paths = tmp_it
    elif paths is not None:
        paths = iter(paths)
    # try to get the filename from  the data store of the obj
    store = getattr(ds, '_file_obj', None)
    store_mod = store.__module__ if store is not None else None
    store_cls = store.__class__.__name__ if store is not None else None
    if store is not None:
        # try several datasets
        for func in get_fname_funcs:
            fname = func(store)
            if fname is not None:
                break
    # check if paths is provided and if yes, save the file
    if fname is None and paths is not None:
        fname = next(paths, None)
        if fname is not None:
            store_mod, store_cls = dump_nc()
    # check create a temporary file with the data if dump is True
    if fname is None:
        fname = NamedTemporaryFile().name + '.nc'
        if not dump and paths is None:
            return None, None, None
        warn('Saving unsaved dataset to %s' % fname)
        store_mod, store_cls = dump_nc()

    return fname, store_mod, store_cls


class CFDecoder(object):
    """
    Class that interpretes the coordinates and attributes accordings to
    cf-conventions"""

    _registry = []

    @property
    def logger(self):
        """:class:`logging.Logger` of this instance"""
        try:
            return self._logger
        except AttributeError:
            name = '%s.%s' % (self.__module__, self.__class__.__name__)
            self._logger = logging.getLogger(name)
            self.logger.debug('Initializing...')
            return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = value

    def __init__(self, ds=None, x=None, y=None, z=None, t=None):
        self.ds = ds
        self.x = rcParams['decoder.x'] if x is None else set(x)
        self.y = rcParams['decoder.y'] if x is None else set(y)
        self.z = rcParams['decoder.z'] if x is None else set(z)
        self.t = rcParams['decoder.t'] if x is None else set(t)

    @staticmethod
    def register_decoder(decoder_class, pos=0):
        """Register a new decoder

        This function registeres a decoder class to use

        Parameters
        ----------
        decoder_class: type
            The class inherited from the :class:`CFDecoder`
        pos: int
            The position where to register the decoder (by default: the first
            position"""
        CFDecoder._registry.insert(pos, decoder_class)

    @classmethod
    @docstrings.get_sectionsf('CFDecoder.can_decode', sections=['Parameters',
                                                                'Returns'])
    def can_decode(cls, ds, var):
        """
        Class method to determine whether the object can be decoded by this
        decoder class.

        Parameters
        ----------
        ds: xarray.Dataset
            The dataset that contains the given `var`
        var: xarray.Variable or xarray.DataArray
            The array to decode

        Returns
        -------
        bool
            True if the decoder can decode the given array `var`. Otherwise
            False

        Notes
        -----
        The default implementation returns True for any argument. Subclass this
        method to be specific on what type of data your decoder can decode
        """
        return True

    @classmethod
    @docstrings.dedent
    def get_decoder(cls, ds, var):
        """
        Class method to get the right decoder class that can decode the
        given dataset and variable

        Parameters
        ----------
        %(CFDecoder.can_decode.parameters)s

        Returns
        -------
        CFDecoder
            The decoder for the given dataset that can decode the variable
            `var`"""
        for decoder_cls in cls._registry:
            if decoder_cls.can_decode(ds, var):
                return decoder_cls(ds)
        return CFDecoder(ds)

    @staticmethod
    @docstrings.get_sectionsf('CFDecoder.decode_coords', sections=[
        'Parameters', 'Returns'])
    def decode_coords(ds, gridfile=None, inplace=True):
        """
        Sets the coordinates and bounds in a dataset

        This static method sets those coordinates and bounds that are marked
        marked in the netCDF attributes as coordinates in :attr:`ds` (without
        deleting them from the variable attributes because this information is
        necessary for visualizing the data correctly)

        Parameters
        ----------
        ds: xarray.Dataset
            The dataset to decode
        gridfile: str
            The path to a separate grid file or a xarray.Dataset instance which
            may store the coordinates used in `ds`
        inplace: bool, optional
            If True, `ds` is modified in place

        Returns
        -------
        xarray.Dataset
            `ds` with additional coordinates"""
        def add_attrs(obj):
            if 'coordinates' in obj.attrs:
                extra_coords.update(obj.attrs['coordinates'].split())
            if 'bounds' in obj.attrs:
                extra_coords.add(obj.attrs['bounds'])
        if gridfile is not None and not isinstance(gridfile, xarray.Dataset):
            gridfile = open_dataset(gridfile)
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
        var: xarray.Variable or xarray.DataArray
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
    def _check_triangular_bounds(self, var, axis='x', nans=None):
        """
        Checks whether the bounds in the variable attribute are triangular

        Parameters
        ----------
        var: xarray.Variable or xarray.DataArray
            The variable to check
        axis: {'x', 'y'}
            The spatial axis to check
        nans: {None, 'skip', 'only'}
            Determines whether values with nan shall be left (None), skipped
            (``'skip'``) or shall be the only one returned (``'only'``)

        Returns
        -------
        bool or None
            True, if unstructered, None if it could not be determined
        xarray.Coordinate or None
            the bounds corrdinate (if existent)"""
        coord = self.get_variable_by_axis(var, axis)
        if coord is not None:
            bounds = coord.attrs.get('bounds')
            if bounds is not None:
                bounds = self.ds.coords.get(bounds)
                if nans == 'skip':
                    bounds = bounds[~np.isnan(var.values)]
                elif nans == 'only':
                    bounds = bounds[np.isnan(var.values)]
                elif nans is None:
                    pass
                else:
                    raise ValueError(
                        "`nans` must be either None, 'skip', or 'only'! "
                        "Not {0}!".format(str(nans)))
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
    def is_circumpolar(self, var):
        """
        Test if a variable is on a circumpolar grid

        Parameters
        ----------
        %(CFDecoder.is_triangular.parameters)s

        Returns
        -------
        %(CFDecoder.is_triangular.returns)s"""
        return self.get_x(var).ndim == 2

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
        var: xarray.Variable
            The variable to get the dimension for
        axis: {'x', 'y', 'z', 't'}
            The axis string that identifies the dimension
        coords: dict
            Coordinates to use. If None, the coordinates of the dataset in the
            :attr:`ds` attribute are used.

        Returns
        -------
        xarray.Coordinate or None
            The coordinate for `var` that matches the given `axis` or None if
            no coordinate with the right `axis` could be found.

        Warning
        -------
        If None of the coordinates have an ``'axis'`` attribute, we use the
        ``'coordinate'`` attribute of `var` (if existent).
        Since however the CF Conventions do not determine the order on how
        the coordinates shall be saved, we try to use a pattern matching
        for latitude (``'lat'``) and longitude (``lon'``). If this patterns
        do not match, we interpret the  coordinates such that x: -1, y: -2,
        z: -3. This is all not very safe for awkward dimension names,
        but works for most cases. If you want to be a hundred percent sure,
        use the :attr:`x`, :attr:`y`, :attr:`z` and :attr:`t` attribute.

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
        # If the coordinates is specified but the coordinate
        # variables themselves have no 'axis' attribute, we interpret the
        # coordinates such that x: -1, y: -2, z: -3
        # Since however the CF Conventions do not determine the order on how
        # the coordinates shall be saved, we try to use a pattern matching
        # for latitude and longitude. This is not very nice, hence it is
        # better to specify the :attr:`x` and :attr:`y` attribute
        tname = next(iter(self.t.intersection(coord_names)), None)
        if axis == 'x':
            for cname in filter(lambda cname: re.search('lon', cname),
                                coord_names):
                return coords[cname]
            return coords.get(coord_names[-1])
        elif axis == 'y' and len(coord_names) >= 2:
            for cname in filter(lambda cname: re.search('lat', cname),
                                coord_names):
                return coords[cname]
            return coords.get(coord_names[-2])
        elif (axis == 'z' and len(coord_names) >= 3 and
              coord_names[-3] != tname):
            return coords.get(coord_names[-3])
        elif axis == 't' and tname:
            return coords.get(tname)

    @docstrings.get_sectionsf("CFDecoder.get_x", sections=[
        'Parameters', 'Returns'])
    @dedent
    def get_x(self, var, coords=None):
        """
        Get the x-coordinate of a variable

        This method searches for the x-coordinate in the :attr:`ds`. It first
        checks whether there is one dimension that holds an ``'axis'``
        attribute with 'X', otherwise it looks whether there is an intersection
        between the :attr:`x` attribute and the variables dimensions, otherwise
        it returns the coordinate corresponding to the last dimension of `var`

        Possible types
        --------------
        var: xarray.Variable
            The variable to get the x-coordinate for
        coords: dict
            Coordinates to use. If None, the coordinates of the dataset in the
            :attr:`ds` attribute are used.

        Returns
        -------
        xarray.Coordinate or None
            The y-coordinate or None if it could be found"""
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
        var: xarray.Variables
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

    @docstrings.get_sectionsf("CFDecoder.get_y", sections=[
        'Parameters', 'Returns'])
    @dedent
    def get_y(self, var, coords=None):
        """
        Get the y-coordinate of a variable

        This method searches for the y-coordinate in the :attr:`ds`. It first
        checks whether there is one dimension that holds an ``'axis'``
        attribute with 'Y', otherwise it looks whether there is an intersection
        between the :attr:`y` attribute and the variables dimensions, otherwise
        it returns the coordinate corresponding to the second last dimension of
        `var` (or the last if the dimension of var is one-dimensional)

        Possible types
        --------------
        var: xarray.Variable
            The variable to get the y-coordinate for
        coords: dict
            Coordinates to use. If None, the coordinates of the dataset in the
            :attr:`ds` attribute are used.

        Returns
        -------
        xarray.Coordinate or None
            The y-coordinate or None if it could be found"""
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
        var: xarray.Variables
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

    @docstrings.get_sectionsf("CFDecoder.get_z", sections=[
        'Parameters', 'Returns'])
    @dedent
    def get_z(self, var, coords=None):
        """
        Get the vertical (z-) coordinate of a variable

        This method searches for the z-coordinate in the :attr:`ds`. It first
        checks whether there is one dimension that holds an ``'axis'``
        attribute with 'Z', otherwise it looks whether there is an intersection
        between the :attr:`z` attribute and the variables dimensions, otherwise
        it returns the coordinate corresponding to the third last dimension of
        `var` (or the second last or last if var is two or one-dimensional)

        Possible types
        --------------
        var: xarray.Variable
            The variable to get the z-coordinate for
        coords: dict
            Coordinates to use. If None, the coordinates of the dataset in the
            :attr:`ds` attribute are used.

        Returns
        -------
        xarray.Coordinate or None
            The z-coordinate or None if no z coordinate could be found"""
        coords = coords or self.ds.coords
        coord = self.get_variable_by_axis(var, 'z', coords)
        if coord is not None:
            return coord
        zname = self.get_zname(var)
        if zname is not None:
            return coords.get(zname)
        return None

    def get_zname(self, var, coords=None):
        """Get the name of the z-dimension

        This method gives the name of the z-dimension (which is not necessarily
        the name of the coordinate if the variable has a coordinate attribute)

        Parameters
        ----------
        var: xarray.Variables
            The variable to get the dimension for
        coords: dict
            The coordinates to use for checking the axis attribute. If None,
            they are not used

        Returns
        -------
        str or None
            The coordinate name or None if no vertical coordinate could be
            found

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
        icheck = -2 if is_unstructured else -3
        min_dim = abs(icheck) if 'variable' not in var.dims else abs(icheck-1)
        if var.ndim >= min_dim and var.dims[icheck] != self.get_tname(
                var, coords):
            return var.dims[icheck]
        return None

    @docstrings.get_sectionsf("CFDecoder.get_t", sections=[
        'Parameters', 'Returns'])
    @dedent
    def get_t(self, var, coords=None):
        """
        Get the time coordinate of a variable

        This method searches for the time coordinate in the :attr:`ds`. It
        first checks whether there is one dimension that holds an ``'axis'``
        attribute with 'T', otherwise it looks whether there is an intersection
        between the :attr:`t` attribute and the variables dimensions, otherwise
        it returns the coordinate corresponding to the first dimension of `var`

        Possible types
        --------------
        var: xarray.Variable
            The variable to get the time coordinate for
        coords: dict
            Coordinates to use. If None, the coordinates of the dataset in the
            :attr:`ds` attribute are used.

        Returns
        -------
        xarray.Coordinate or None
            The time coordinate or None if no time coordinate could be found"""
        coords = coords or self.ds.coords
        coord = self.get_variable_by_axis(var, 't', coords)
        if coord is not None:
            return coord
        dimlist = list(self.t.intersection(var.dims).intersection(coords))
        if dimlist:
            if len(dimlist) > 1:
                warn("Found multiple matches for time coordinate in the "
                     "variable: %s. I use %s" % (
                         ', '.join(dimlist), dimlist[0]),
                     PsyPlotRuntimeWarning)
            return coords[dimlist[0]]
        tname = self.get_tname(var)
        if tname is not None:
            return coords.get(tname)
        return None

    def get_tname(self, var, coords=None):
        """Get the name of the t-dimension

        This method gives the name of the time dimension

        Parameters
        ----------
        var: xarray.Variables
            The variable to get the dimension for
        coords: dict
            The coordinates to use for checking the axis attribute. If None,
            they are not used

        Returns
        -------
        str or None
            The coordinate name or None if no time coordinate could be found

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
        # otherwise we return None
        return None

    def get_idims(self, arr, coords=None):
        """Get the coordinates in the :attr:`ds` dataset as int or slice

        This method returns a mapping from the coordinate names of the given
        `arr` to an integer, slice or an array of integer that represent the
        coordinates in the :attr:`ds` dataset and can be used to extract the
        given `arr` via the :meth:`xarray.Dataset.isel` method.

        Parameters
        ----------
        arr: xarray.DataArray
            The data array for which to get the dimensions as integers, slices
            or list of integers from the dataset in the :attr:`base` attribute

        Returns
        -------
        dict
            Mapping from coordinate name to integer, list of integer or slice

        See Also
        --------
        xarray.Dataset.isel, InteractiveArray.idims"""
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
        coord: xarray.Coordinate
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
                bounds = self.ds.isel(**self.get_idims(coord))
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
        coord: xarray.Coordinate
            The coordinate to get the bounds for
        bounds: xarray.DataArray
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
        ret = np.zeros(tuple(map(lambda i: i+1, coord.shape)))
        ret[tuple(map(slice, coord.shape))] = bounds[..., 0]
        last_slices = tuple(slice(-1, None) for _ in coord.shape)
        ret[last_slices] = bounds[tuple(chain(last_slices, [1]))]
        return ret

    docstrings.keep_params('CFDecoder._check_triangular_bounds.parameters',
                           'nans')

    @docstrings.get_sectionsf('CFDecoder.get_triangles', sections=[
        'Parameters', 'Returns'])
    @docstrings.dedent
    def get_triangles(self, var, coords=None, convert_radian=True,
                      copy=False, src_crs=None, target_crs=None,
                      nans=None):
        """
        Get the triangles for the variable

        Parameters
        ----------
        var: xarray.Variable or xarray.DataArray
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
        %(CFDecoder._check_triangular_bounds.parameters.nans)s

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
            bounds = self._check_triangular_bounds(var, axis, nans=nans)[1]
            if coords is not None:
                bounds = coords.get(bounds.name, bounds)
            vertices = bounds.values.ravel()
            if convert_radian:
                coord = getattr(self, 'get_' + axis)(var)
                if coord.attrs.get('units') == 'radian':
                    vertices = vertices * 180. / np.pi
            return vertices if not copy else vertices.copy()

        if coords is None:
            coords = self.ds.coords

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
        triangles = np.reshape(range(len(xvert)), (len(xvert) / 3, 3))
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
            y, x = map(np.arange, coord.shape)
            new_x, new_y = map(_infer_interval_breaks, [x, y])
            return interp2d(x, y, np.asarray(coord), kind=kind)(new_x, new_y)

    @classmethod
    @docstrings.get_sectionsf('CFDecoder._decode_ds')
    @docstrings.dedent
    def _decode_ds(cls, ds, gridfile=None, inplace=False, decode_coords=True,
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
            ds = cls.decode_coords(ds, gridfile=gridfile,
                                   inplace=inplace)
        if decode_times:
            for k, v in six.iteritems(ds.variables):
                # check for absolute time units and make sure the data is not
                # already decoded via dtype check
                if v.attrs.get('units', '') == 'day as %Y%m%d.%f' and (
                        np.issubdtype(v.dtype, float)):
                    decoded = xarray.Variable(
                        v.dims, AbsoluteTimeDecoder(v), attrs=v.attrs,
                        encoding=v.encoding)
                    ds = ds.update({k: decoded}, inplace=inplace)
        return ds

    @classmethod
    @docstrings.dedent
    def decode_ds(cls, ds, *args, **kwargs):
        """
        Static method to decode coordinates and time informations

        This method interpretes absolute time informations (stored with units
        ``'day as %Y%m%d.%f'``) and coordinates

        Parameters
        ----------
        %(CFDecoder._decode_ds.parameters)s

        Returns
        -------
        xarray.Dataset
            The decoded dataset"""
        for decoder_cls in cls._registry + [CFDecoder]:
            ds = decoder_cls._decode_ds(ds, *args, **kwargs)
        return ds

    def correct_dims(self, var, dims={}):
        """Expands the dimensions to match the dims in the variable

        Parameters
        ----------
        var: xarray.Variable
            The variable to get the data for
        dims: dict
            a mapping from dimension to the slices"""
        method_mapping = {'x': self.get_xname,
                          'z': self.get_zname, 't': self.get_tname}
        dims = dict(dims)
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
                    new_name = method_mapping[key](var)
                    if new_name is not None:
                        dims[new_name] = dims.pop(key)
        # now remove the unnecessary dimensions
        for key in set(dims).difference(var.dims):
            dims.pop(key)
            self.logger.debug(
                "Could not find a dimensions matching %s in variable %s!",
                key, var)
        return dims

    def standardize_dims(self, var, dims={}):
        """Replace the coordinate names through x, y, z and t

        Parameters
        ----------
        var: xarray.Variable
            The variable to use the dimensions of
        dims: dict
            The dictionary to use for replacing the original dimensions"""
        dims = dict(dims)
        name_map = {self.get_xname(var, self.ds.coords): 'x',
                    self.get_yname(var, self.ds.coords): 'y',
                    self.get_zname(var, self.ds.coords): 'z',
                    self.get_tname(var, self.ds.coords): 't'}
        dims = dict(dims)
        for dim in set(dims).intersection(name_map):
            dims[name_map[dim]] = dims.pop(dim)
        return dims


class UGridDecoder(CFDecoder):
    """
    Decoder for UGrid data sets

    Warnings
    --------
    Currently only triangles are supported."""

    def is_triangular(self, *args, **kwargs):
        """Reimpletemented to return always True. Any ``*args`` and ``**kwargs``
        are ignored"""
        return True

    def get_mesh(self, var, coords=None):
        """Get the mesh variable for the given `var`

        Parameters
        ----------
        var: xarray.Variable
            The data source whith the ``'mesh'`` attribute
        coords: dict
            The coordinates to use. If None, the coordinates of the dataset of
            this decoder is used

        Returns
        -------
        xarray.Coordinate
            The mesh coordinate"""
        mesh = var.attrs.get('mesh')
        if mesh is None:
            return None
        if coords is None:
            coords = self.ds.coords
        return coords.get(mesh, self.ds.coords.get(mesh))

    @classmethod
    @docstrings.dedent
    def can_decode(cls, ds, var):
        """
        Check whether the given variable can be decoded.

        Returns True if a mesh coordinate could be found via the
        :meth:`get_mesh` method

        Parameters
        ----------
        %(CFDecoder.can_decode.parameters)%s

        Returns
        -------
        %(CFDecoder.can_decode.returns)%s"""
        return cls(ds).get_mesh(var) is not None

    @docstrings.dedent
    def get_triangles(self, var, coords=None, convert_radian=True, copy=False,
                      src_crs=None, target_crs=None, nans=None):
        """
        Get the of the given coordinate.

        Parameters
        ----------
        %(CFDecoder.get_triangles.parameters)s

        Returns
        -------
        %(CFDecoder.get_triangles.returns)s

        Notes
        -----
        If the ``'location'`` attribute is set to ``'node'``, a delaunay
        triangulation is performed using the
        :class:`matplotlib.tri.Triangulation` class.

        .. todo::
            Implement the visualization for UGrid data shown on the edge of the
            triangles"""
        from matplotlib.tri import Triangulation

        if coords is None:
            coords = self.ds.coords

        def get_coord(coord):
            return coords.get(coord, self.ds.coords.get(coord))

        mesh = self.get_mesh(var, coords)
        nodes = self.get_nodes(mesh, coords)
        if any(n is None for n in nodes):
            raise ValueError("Could not find the nodes variables!")
        xvert, yvert = nodes
        xvert = xvert.values
        yvert = yvert.values
        loc = var.attrs.get('location', 'face')
        if loc == 'face':
            triangles = get_coord(
                mesh.attrs.get('face_node_connectivity', '')).values
            if triangles is None:
                raise ValueError(
                    "Could not find the connectivity information!")
        elif loc == 'node':
            triangles = None
        else:
            raise ValueError(
                "Could not interprete location attribute (%s) of mesh "
                "variable %s!" % (loc, mesh.name))

        if convert_radian:
            for coord in nodes:
                if coord.attrs.get('units') == 'radian':
                    coord = coord * 180. / np.pi
        if src_crs is not None and src_crs != target_crs:
            if target_crs is None:
                raise ValueError(
                    "Found %s for the source crs but got None for the "
                    "target_crs!" % (src_crs, ))
            xvert = xvert[triangles].ravel()
            yvert = yvert[triangles].ravel()
            arr = target_crs.transform_points(src_crs, xvert, yvert)
            xvert = arr[:, 0]
            yvert = arr[:, 1]
            if loc == 'face':
                triangles = np.reshape(range(len(xvert)), (len(xvert) / 3, 3))

        return Triangulation(xvert, yvert, triangles)

    @staticmethod
    @docstrings.dedent
    def decode_coords(ds, gridfile=None, inplace=True):
        """
        Reimplemented to set the mesh variables as coordinates

        Parameters
        ----------
        %(CFDecoder.decode_coords.parameters)s

        Returns
        -------
        %(CFDecoder.decode_coords.returns)s"""
        extra_coords = set(ds.coords)
        for var in six.itervalues(ds.variables):
            if 'mesh' in var.attrs:
                mesh = var.attrs['mesh']
                if mesh not in extra_coords:
                    extra_coords.add(mesh)
                    try:
                        mesh_var = ds.variables[mesh]
                    except KeyError:
                        warn('Could not find mesh variable %s' % mesh)
                        continue
                    if 'node_coordinates' in mesh_var.attrs:
                        extra_coords.update(
                            mesh_var.attrs['node_coordinates'].split())
                    if 'face_node_connectivity' in mesh_var.attrs:
                        extra_coords.add(
                            mesh_var.attrs['face_node_connectivity'])
        if gridfile is not None and not isinstance(gridfile, xarray.Dataset):
            gridfile = open_dataset(gridfile)
            ds = ds.update({k: v for k, v in six.iteritems(gridfile.variables)
                            if k in extra_coords}, inplace=inplace)
        ds = ds.set_coords(extra_coords.intersection(ds.variables),
                           inplace=inplace)
        return ds

    def get_nodes(self, coord, coords):
        """Get the variables containing the definition of the nodes

        Parameters
        ----------
        coord: xarray.Coordinate
            The mesh variable
        coords: dict
            The coordinates to use to get node coordinates"""
        def get_coord(coord):
            return coords.get(coord, self.ds.coords.get(coord))
        return list(map(get_coord,
                        coord.attrs.get('node_coordinates', '').split()[:2]))

    @docstrings.dedent
    def get_x(self, var, coords=None):
        """Get the centers of the triangles in the x-dimension

        Parameters
        ----------
        %(CFDecoder.get_y.parameters)s

        Returns
        -------
        %(CFDecoder.get_y.returns)s"""
        if coords is None:
            coords = self.ds.coords
        # first we try the super class
        ret = super(UGridDecoder, self).get_x(var, coords)
        # but if that doesn't work because we get the variable name in the
        # dimension of `var`, we use the means of the triangles
        if ret is None or ret.name in var.dims:
            triangles = self.get_triangles(var, coords)
            centers = triangles.x[triangles.triangles].mean(axis=-1)
            x = self.get_nodes(self.get_mesh(var, coords), coords)[0]
            return xarray.Coordinate(x.name, centers, attrs=x.attrs.copy())

    @docstrings.dedent
    def get_y(self, var, coords=None):
        """
        Get the centers of the triangles in the y-dimension

        Parameters
        ----------
        %(CFDecoder.get_y.parameters)s

        Returns
        -------
        %(CFDecoder.get_y.returns)s"""
        if coords is None:
            coords = self.ds.coords
        # first we try the super class
        ret = super(UGridDecoder, self).get_y(var, coords)
        # but if that doesn't work because we get the variable name in the
        # dimension of `var`, we use the means of the triangles
        if ret is None or ret.name in var.dims:
            triangles = self.get_triangles(var, coords)
            centers = triangles.y[triangles.triangles].mean(axis=-1)
            y = self.get_nodes(self.get_mesh(var, coords), coords)[1]
            return xarray.Coordinate(y.name, centers, attrs=y.attrs.copy())


# register the UGridDecoder
CFDecoder.register_decoder(UGridDecoder)

docstrings.keep_params('CFDecoder.decode_coords.parameters', 'gridfile')
docstrings.get_sections(dedents(xarray.open_dataset.__doc__.split('\n', 1)[1]),
                        'xarray.open_dataset')
docstrings.delete_params('xarray.open_dataset.parameters', 'engine')


@docstrings.get_sectionsf('open_dataset')
@docstrings.dedent
def open_dataset(filename_or_obj, decode_cf=True, decode_times=True,
                 decode_coords=True, engine=None, gridfile=None, **kwargs):
    """
    Open an instance of :class:`xarray.Dataset`.

    This method has the same functionality as the :func:`xarray.open_dataset`
    method except that is supports an additional 'gdal' engine to open
    gdal Rasters (e.g. GeoTiffs) and that is supports absolute time units like
    ``'day as %Y%m%d.%f'`` (if `decode_cf` and `decode_times` are True).

    Parameters
    ----------
    %(xarray.open_dataset.parameters.no_engine)s
    engine: {'netcdf4', 'scipy', 'pydap', 'h5netcdf', 'gdal'}, optional
        Engine to use when reading netCDF files. If not provided, the default
        engine is chosen based on available dependencies, with a preference for
        'netcdf4'.
    %(CFDecoder.decode_coords.parameters.gridfile)s

    Returns
    -------
    xarray.Dataset
        The dataset that contains the variables from `filename_or_obj`"""
    # use the absolute path name (is saver when saving the project)
    if isstring(filename_or_obj) and os.path.exists(filename_or_obj):
        filename_or_obj = os.path.abspath(filename_or_obj)
    if engine == 'gdal':
        from .gdal_store import GdalStore
        filename_or_obj = GdalStore(filename_or_obj)
        engine = None
    ds = xarray.open_dataset(filename_or_obj, decode_cf=decode_cf,
                           decode_coords=False, engine=engine,
                           decode_times=decode_times, **kwargs)
    if decode_cf:
        ds = CFDecoder.decode_ds(
            ds, decode_coords=decode_coords, decode_times=decode_times,
            gridfile=gridfile, inplace=True)
    return ds


docstrings.get_sections(dedents(xarray.open_mfdataset.__doc__.split('\n', 1)[1]),
                        'xarray.open_mfdataset')
docstrings.delete_params('xarray.open_mfdataset.parameters', 'engine')
docstrings.keep_params('get_tdata.parameters', 't_format')

docstrings.params['xarray.open_mfdataset.parameters.no_engine'] = \
    docstrings.params['xarray.open_mfdataset.parameters.no_engine'].replace(
        '**kwargs', '``**kwargs``').replace('"path/to/my/files/*.nc"',
                                            '``"path/to/my/files/*.nc"``')


docstrings.keep_params('open_dataset.parameters', 'engine')


@docstrings.dedent
def open_mfdataset(paths, decode_cf=True, decode_times=True,
                   decode_coords=True, engine=None, gridfile=None,
                   t_format=None, **kwargs):
    """
    Open multiple files as a single dataset.

    This function is essentially the same as the :func:`xarray.open_mfdataset`
    function but (as the :func:`open_dataset`) supports additional decoding
    and the ``'gdal'`` engine.
    You can further specify the `t_format` parameter to get the time
    information from the files and use the results to concatenate the files

    Parameters
    ----------
    %(xarray.open_mfdataset.parameters.no_engine)s
    %(open_dataset.parameters.engine)s
    %(get_tdata.parameters.t_format)s
    %(CFDecoder.decode_coords.parameters.gridfile)s

    Returns
    -------
    xarray.Dataset
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

    ds = xarray.open_mfdataset(
        paths, decode_cf=decode_cf, decode_times=decode_times, engine=engine,
        decode_coords=False, **kwargs)
    if decode_cf:
        return CFDecoder.decode_ds(ds, gridfile=gridfile, inplace=True,
                                   decode_coords=decode_coords,
                                   decode_times=decode_times)
    return ds


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

    no_auto_update = property(_no_auto_update_getter,
                              doc=_no_auto_update_getter.__doc__)

    @no_auto_update.setter
    def no_auto_update(self, value):
        if self.plotter is not None:
            self.plotter.no_auto_update = value
        self.no_auto_update.value = bool(value)

    @property
    def logger(self):
        """:class:`logging.Logger` of this instance"""
        try:
            return self._logger
        except AttributeError:
            name = '%s.%s.%s' % (self.__module__, self.__class__.__name__,
                                 self.arr_name)
            self._logger = logging.getLogger(name)
            self.logger.debug('Initializing...')
            return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = value

    @property
    def ax(self):
        """The matplotlib axes the plotter of this data object plots on"""
        return None if self.plotter is None else self.plotter.ax

    @ax.setter
    def ax(self, value):
        if self.plotter is None:
            raise ValueError(
                'Cannot set the axes because the plotter attribute is None!')
        self.plotter.ax = value

    # -------------------------------------------------------------------------
    # -------------------------------- SIGNALS --------------------------------
    # -------------------------------------------------------------------------

    #: :class:`Signal` to be emitted when the object has been updated
    onupdate = Signal('_onupdate')
    _onupdate = None

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

    _no_auto_update = None

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
            :meth:`update` method or not. See also the :attr:`no_auto_update`
            attribute. If None, the value from the ``'lists.auto_update'``
            key in the :attr:`psyplot.rcParams` dictionary is used."""
        self.plotter = plotter
        self.arr_name = arr_name
        if auto_update is None:
            auto_update = rcParams['lists.auto_update']
        self.auto_update = not bool(auto_update)
        self.replot = False

    def _finish_all(self, queues):
        for n, queue in zip(safe_list(self._njobs), safe_list(queues)):
            if queue is not None:
                for i in range(n):
                    queue.task_done()

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
    def start_update(self, draw=None, queues=None):
        """
        Conduct the formerly registered updates

        This method conducts the updates that have been registered via the
        :meth:`update` method. You can call this method if the
        :attr:`no_auto_update` attribute of this instance and the `auto_update`
        parameter in the :meth:`update` method has been set to False

        Parameters
        ----------
        draw: bool or None
            Boolean to control whether the figure of this array shall be drawn
            at the end. If None, it defaults to the `'auto_draw'`` parameter
            in the :attr:`psyplot.rcParams` dictionary
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
        :attr:`no_auto_update`, update
        """
        if self.plotter is not None:
            return self.plotter.start_update(draw=draw, queues=queues)

    docstrings.keep_params('InteractiveBase.start_update.parameters', 'draw')

    @docstrings.get_sectionsf('InteractiveBase.update',
                              sections=['Parameters', 'Notes'])
    @docstrings.dedent
    def update(self, fmt={}, replot=False, draw=None, auto_update=False,
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
            is called at the end. This parameter has no effect if the
            :attr:`no_auto_update` attribute is set to ``True``.
        %(InteractiveBase.start_update.parameters.draw)s
        ``**kwargs``
            Any other formatoption that shall be updated (additionally to those
            in `fmt`)

        Notes
        -----
        If the :attr:`no_auto_update` attribute is True and the given
        `auto_update` parameter are is False, the update of the plots are
        registered and conducted at the next call of the :meth:`start_update`
        method or the next call of this method (if the `auto_update` parameter
        is then True).
        """
        fmt = dict(fmt)
        fmt.update(kwargs)

        self._register_update(replot=replot, fmt=fmt, force=force,
                              todefault=todefault)

        if not self.no_auto_update or auto_update:
            self.start_update(draw=draw)


class InteractiveArray(xarray.DataArray, InteractiveBase):
    """Interactive version of the :class:`xarray.DataArray`

    This class keeps reference to the base :class:`xarray.Dataset` where it
    originates from and enables to switch between the coordinates in this
    array. Furthermore it has a :attr:`plotter` attribute to enable interactive
    plotting via an :class:`psyplot.plotter.Plotter` instance."""

    @property
    def base(self):
        """Base dataset this instance gets its data from"""
        if self._base is None:
            if 'variable' in self.dims:
                def to_dataset(i):
                    return self.isel(variable=i).drop('variable').to_dataset(
                        name=self.coords['variable'].values[i])
                ds = to_dataset(0)
                if len(self.coords['variable']) > 1:
                    for i in range(1, len(self.coords['variable'])):
                        ds.merge(to_dataset(i), inplace=True)
                self._base = ds
            else:
                self._base = self.to_dataset(name=self.name or self.arr_name)
            self.onbasechange.emit()
        return self._base

    @base.setter
    def base(self, value):
        self._base = value
        self.onbasechange.emit()

    @property
    def decoder(self):
        """The decoder of this array"""
        return getattr(self, '_decoder', CFDecoder.get_decoder(
            self.base, self))

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

    logger = InteractiveBase.logger
    _idims = None
    _base = None

    # -------------------------------------------------------------------------
    # -------------------------------- SIGNALS --------------------------------
    # -------------------------------------------------------------------------
    #: :class:`Signal` to be emiited when the base of the object changes
    onbasechange = Signal('_onbasechange')
    _onbasechange = None

    @docstrings.dedent
    def __init__(self, *args, **kwargs):
        """
        The ``*args`` and ``**kwargs`` are essentially the same as for the
        :class:`xarray.DataArray` method, additional ``**kwargs`` are
        described below.

        Other Parameters
        ----------------
        base: xarray.Dataset
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
        xarray.DataArray.__init__(self, *args, **kwargs)

    @classmethod
    def _new_from_dataset_no_copy(cls, *args, **kwargs):
        obj = super(cls, cls)._new_from_dataset_no_copy(*args, **kwargs)
        obj._registered_updates = {}
        obj._new_dims = {}
        obj.method = None
        obj.arr_name = 'arr'
        obj.no_auto_update = not bool(kwargs.pop(
            'auto_update', rcParams['lists.auto_update']))
        obj.replot = False
        return obj

    def copy(self, *args, **kwargs):
        arr_name = kwargs.pop('arr_name', self.arr_name)
        obj = super(InteractiveArray, self).copy(*args, **kwargs)
        obj.no_auto_update = bool(self.no_auto_update)
        obj.arr_name = arr_name
        obj.replot = self.replot
        return obj

    copy.__doc__ = xarray.DataArray.copy.__doc__ + """
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
    def iter_base_variables(self):
        """An iterator over the base variables in the :attr:`base` dataset"""
        if 'variable' in self.coords:
            return (self.base.variables[name] for name in safe_list(
                self.coords['variable'].values.tolist()))
        return iter([self.base.variables[self.name]])

    @property
    def base_variables(self):
        """A mapping from the variable name to the variablein the :attr:`base`
        dataset."""
        if 'variable' in self.coords:
            return OrderedDict([(name, self.base.variables[name])
                                for name in safe_list(
                                    self.coords['variable'].values.tolist())])
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
            Selection method of the xarray.Dataset to be used for setting the
            variables from the informations in `dims`.
            If `method` is 'isel', the :meth:`xarray.Dataset.isel` method is
            used. Otherwise it sets the `method` parameter for the
            :meth:`xarray.Dataset.sel` method.
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
        if 'name' in dims:
            self._new_dims['name'] = dims.pop('name')
        self._new_dims.update(self.decoder.correct_dims(
            next(six.itervalues(self.base_variables)), dims))
        InteractiveBase._register_update(
            self, fmt=fmt, replot=replot or bool(self._new_dims), force=force,
            todefault=todefault)

    def _update_concatenated(self, dims, method):
        """Updates a concatenated array to new dimensions"""
        def filter_attrs(item):
            """Checks whether the attribute is from the :attr:`base` dataset"""
            return (item[0] not in self.base.attrs or
                    item[1] != self.base.attrs[item[0]])
        saved_attrs = list(filter(filter_attrs, six.iteritems(self.attrs)))
        saved_name = self.name
        self.name = 'None'
        if 'name' in dims:
            name = dims.pop('name')
        else:
            name = list(self.coords['variable'].values)
        if method == 'isel':
            self.idims.update(dims)
            dims = self.idims
            res = self.base[name].isel(**dims).to_array()
        else:
            for key, val in six.iteritems(self.coords):
                if key != 'variable':
                    dims.setdefault(key, val)
            if any(isinstance(idx, slice) for idx in dims.values()):
                # ignore method argument
                res = self.base[name].sel(**dims).to_array()
            else:
                res = self.base[name].sel(method=method, **dims).to_array()
        self._variable = res._variable
        self._coords = res._coords
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
            res = self.base[name].isel(**dims)
        else:
            for key, val in six.iteritems(self.coords):
                dims.setdefault(key, val)
            if any(isinstance(idx, slice) for idx in dims.values()):
                # ignore method argument
                res = self.base[name].sel(**dims)
            else:
                res = self.base[name].sel(method=method, **dims)
        self._variable = res._variable
        self._coords = res._coords
        # update to old attributes
        for key, val in saved_attrs:
            self.attrs[key] = val

    @docstrings.dedent
    def start_update(self, draw=None, queues=None):
        """
        Conduct the formerly registered updates

        This method conducts the updates that have been registered via the
        :meth:`update` method. You can call this method if the
        :attr:`no_auto_update` attribute of this instance is True and the
        `auto_update` parameter in the :meth:`update` method has been set to
        False

        Parameters
        ----------
        %(InteractiveBase.start_update.parameters)s

        Returns
        -------
        %(InteractiveBase.start_update.returns)s

        See Also
        --------
        :attr:`no_auto_update`, update
        """
        def filter_attrs(item):
            return (item[0] not in self.base.attrs or
                    item[1] != self.base.attrs[item[0]])
        if queues is not None:
            # make sure that no plot is updated during gathering the data
            queues[0].get()
        try:
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
            self.onupdate.emit()
        except:
            self._finish_all(queues)
            raise
        return InteractiveBase.start_update(self, draw=draw, queues=queues)

    @docstrings.get_sectionsf('InteractiveArray.update',
                              sections=['Parameters', 'Notes'])
    @docstrings.dedent
    def update(self, method='isel', dims={}, fmt={}, replot=False,
               auto_update=False, draw=None, force=False, todefault=False,
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
            self.dims, self.coords, ['name', 'x', 'y', 'z', 't']))
        furtherdims, furtherfmt = sort_kwargs(kwargs, vars_and_coords)
        dims.update(furtherdims)
        fmt.update(furtherfmt)

        self._register_update(method=method, replot=replot, dims=dims,
                              fmt=fmt, force=force, todefault=todefault)

        if not self.no_auto_update or auto_update:
            self.start_update(draw=draw)

    def _short_info(self, intend=0, maybe=False):
        str_intend = ' ' * intend
        if 'variable' in self.coords:
            name = ', '.join(self.coords['variable'].values)
        else:
            name = self.name
        return str_intend + "%s: %s of %s, %s" % (
            self.arr_name, self.__class__.__name__, name, ", ".join(
                "%s: %s" % (coord, format_item(val.values))
                for coord, val in six.iteritems(self.coords) if val.ndim == 0))

    def isel(self, *args, **kwargs):
        # reimplemented to keep the base. The doc is set below
        ret = super(InteractiveArray, self).isel(*args, **kwargs)
        ret._base = self._base
        return ret

    isel.__doc__ = xarray.DataArray.isel.__doc__


class ArrayList(list):
    """Base class for creating a list of interactive arrays from a dataset

    This list contains and manages :class:`InteractiveArray` instances"""

    docstrings.keep_params('InteractiveBase.parameters', 'auto_update')

    @property
    def dims(self):
        """Dimensions of the arrays in this list"""
        return set(chain(*(arr.dims for arr in self)))

    @property
    def dims_intersect(self):
        """Dimensions of the arrays in this list that are used in all arrays
        """
        return set.intersection(*map(
            set, (getattr(arr, 'dims_intersect', arr.dims) for arr in self)))

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
    def coords_intersect(self):
        """Coordinates of the arrays in this list that are used in all arrays
        """
        return set.intersection(*map(
            set, (getattr(arr, 'coords_intersect', arr.coords) for arr in self)
            ))

    @property
    def with_plotter(self):
        """The arrays in this instance that are visualized with a plotter"""
        return self.__class__((arr for arr in self if arr.plotter is not None),
                              auto_update=bool(self.auto_update))

    no_auto_update = property(_no_auto_update_getter,
                              doc=_no_auto_update_getter.__doc__)

    @no_auto_update.setter
    def no_auto_update(self, value):
        for arr in self:
            arr.no_auto_update = value
        self.no_auto_update.value = bool(value)

    @property
    def logger(self):
        """:class:`logging.Logger` of this instance"""
        try:
            return self._logger
        except AttributeError:
            name = '%s.%s' % (self.__module__, self.__class__.__name__)
            self._logger = logging.getLogger(name)
            self.logger.debug('Initializing...')
            return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = value

    @property
    def arrays(self):
        """An iterator over all the data arrays instances in this list"""
        return ArrayList(chain(
            *[[arr] if not isinstance(arr, InteractiveList) else arr.arrays
              for arr in self]))

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
        self.auto_update = not bool(auto_update)

    def copy(self, deep=False):
        """Returns a copy of the list

        Parameters
        ----------
        deep: bool
            If False (default), only the list is copied and not the contained
            arrays, otherwise the contained arrays are deep copied"""
        if not deep:
            return self.__class__(self[:], attrs=self.attrs.copy(),
                                  auto_update=not bool(self.no_auto_update))
        else:
            return self.__class__(
                [arr.copy(deep) for arr in self], attrs=self.attrs.copy(),
                auto_update=not bool(self.auto_update))

    docstrings.keep_params('InteractiveArray.update.parameters', 'method')

    @classmethod
    @docstrings.get_sectionsf('ArrayList.from_dataset', sections=[
        'Parameters', 'Other Parameters', 'Returns'])
    @docstrings.dedent
    def from_dataset(cls, base, method='isel', default_slice=None,
                     decoder=None, auto_update=None, prefer_list=False,
                     squeeze=True, attrs=None, **kwargs):
        """
        Construct an ArrayDict instance from an existing base dataset

        Parameters
        ----------
        base: xarray.Dataset
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
        squeeze: bool, optional
            Default True. If True, and the created arrays have a an axes with
            length 1, it is removed from the dimension list (e.g. an array
            with shape (3, 4, 1, 5) will be squeezed to shape (3, 4, 5))
        attrs: dict, optional
            Meta attributes that shall be assigned to the selected data arrays
            (additional to those stored in the `base` dataset)

        Other Parameters
        ----------------
        %(setup_coords.parameters)s

        Returns
        -------
        ArrayList
            The list with the specified :class:`InteractiveArray` instances
            that hold a reference to the given `base`"""
        def iter_dims(dims):
            """Split the given dictionary into multiples and iterate over it"""
            dims = OrderedDict(dims)
            keys = dims.keys()
            for vals in zip(*map(cycle, map(safe_list, dims.values()))):
                yield dict(zip(keys, vals))

        def recursive_selection(key, dims, names):
            names = safe_list(names)
            if len(names) > 1 and prefer_list:
                keys = ('arr%i' % i for i in range(len(names)))
                return InteractiveList(starmap(
                    sel_method, zip(keys, iter_dims(dims), names)),
                    auto_update=auto_update, arr_name=key)
            elif len(names) > 1:
                return sel_method(key, dims, tuple(names))
            else:
                return sel_method(key, dims, names[0])

        if decoder is not None:
            def get_decoder(arr):
                return decoder
        else:
            def get_decoder(arr):
                return CFDecoder.get_decoder(base, arr)

        if squeeze:
            def squeeze_array(arr):
                return arr.isel(**{dim: 0 for i, dim in enumerate(arr.dims)
                                   if arr.shape[i] == 1})
        else:
            def squeeze_array(arr):
                return arr
        if method == 'isel':
            def sel_method(key, dims, name=None):
                if name is None:
                    return recursive_selection(key, dims, dims.pop('name'))
                elif isinstance(name, six.string_types):
                    arr = base[name]
                else:
                    arr = base[list(name)]
                if not isinstance(arr, xarray.DataArray):
                    attrs = next(var for key, var in arr.variables.items()
                                 if key not in arr.coords).attrs
                    arr = arr.to_array()
                    arr.attrs.update(attrs)
                def_slice = slice(None) if default_slice is None else \
                    default_slice
                decoder = get_decoder(arr)
                dims = decoder.correct_dims(arr, dims)
                dims.update({
                    dim: def_slice for dim in set(arr.dims).difference(
                        dims) if dim != 'variable'})
                return InteractiveArray(squeeze_array(
                    arr.isel(**dims)), arr_name=key, base=base, idims=dims)
        else:
            def sel_method(key, dims, name=None):
                if name is None:
                    return recursive_selection(key, dims, dims.pop('name'))
                elif isinstance(name, six.string_types):
                    arr = base[name]
                else:
                    arr = base[list(name)]
                if not isinstance(arr, xarray.DataArray):
                    attrs = next(var for key, var in arr.variables.items()
                                 if key not in arr.coords).attrs
                    arr = arr.to_array()
                    arr.attrs.update(attrs)
                # idims will be calculated by the array (maybe not the most
                # efficient way...)
                decoder = get_decoder(arr)
                dims = decoder.correct_dims(arr, dims)
                if default_slice is not None:
                    dims.update({
                        key: default_slice for key in set(arr.dims).difference(
                            dims) if key != 'variable'})
                # the sel method does not work with slice objects
                if any(isinstance(idx, slice) for idx in dims.values()):
                    # ignore method argument
                    return InteractiveArray(squeeze_array(
                        arr.sel(**dims)), arr_name=key, base=base)
                else:
                    return InteractiveArray(squeeze_array(
                        arr.sel(method=method, **dims)), arr_name=key,
                        base=base)
        kwargs.setdefault(
            'name', sorted(
                key for key in base.variables if key not in base.coords))
        names = setup_coords(**kwargs)
        # check coordinates
        possible_keys = ['t', 'x', 'y', 'z', 'name'] + list(base.coords)
        for key in set(chain(*six.itervalues(names))):
            check_key(key, possible_keys, name='dimension')
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
        if attrs is not None:
            for arr in instance:
                arr.attrs.update(attrs)
        return instance

    @classmethod
    def _get_dsnames(cls, data, ignore_keys=['attrs', 'plotter', 'ds']):
        """Recursive method to get all the file names out of a dictionary
        `data` created with the :meth`array_info` method"""
        if 'fname' in data:
            return {(data['fname'], data['store'])}
        for key in ignore_keys:
            data.pop(key, None)
        return set(chain(*map(cls._get_dsnames, six.itervalues(data))))

    @classmethod
    def _get_ds_descriptions(
            cls, data, ds_description={'ds', 'fname', 'arr'}, **kwargs):
        def new_dict():
            return defaultdict(list)
        ret = defaultdict(new_dict)
        ds_description = set(ds_description)
        for d in cls._get_ds_descriptions_unsorted(data, **kwargs):
            try:
                num = d.get('num') or cls._get_psyplot_num(d['ds'])
            except KeyError:
                raise ValueError(
                    'Could not find either the dataset number nor the dataset '
                    'in the data! However one must be provided.')
            d_ret = ret[num]
            for key, val in six.iteritems(d):
                if key == 'arr':
                    d_ret['arr'].append(d['arr'])
                else:
                    d_ret[key] = val
        return ret

    @classmethod
    def _get_ds_descriptions_unsorted(
            cls, data, ignore_keys=['attrs', 'plotter'], nums=None):
        """Recursive method to get all the file names or datasets out of a
        dictionary `data` created with the :meth`array_info` method"""
        ds_description = {'ds', 'fname', 'num', 'arr', 'store'}
        if 'ds' in data:
            # make sure that the data set has a number assigned to it
            cls._get_psyplot_num(data['ds'])
        keys_in_data = ds_description.intersection(data)
        if keys_in_data:
            return {key: data[key] for key in keys_in_data}
        for key in ignore_keys:
            data.pop(key, None)
        func = partial(cls._get_ds_descriptions_unsorted,
                       ignore_keys=ignore_keys, nums=nums)
        return chain(*map(lambda d: [d] if isinstance(d, dict) else d,
                          map(func, six.itervalues(data))))

    @classmethod
    @docstrings.get_sectionsf('ArrayList.from_dict')
    @docstrings.dedent
    def from_dict(cls, d, alternative_paths={}, datasets=None,
                  pwd=None, ignore_keys=['attrs', 'plotter', 'ds'], **kwargs):
        """
        Create a list from the dictionary returned by :meth:`array_info`

        This classmethod creates an :class:`~psyplot.data.ArrayList` instance
        from a dictionary containing filename, dimension infos and array names

        Parameters
        ----------
        d: dict
            The dictionary holding the data
        alternative_paths: dict or list or str
            A mapping from original filenames as used in `d` to filenames that
            shall be used instead. If `alternative_paths` is not None,
            datasets must be None. Paths must be accessible from the current
            working directory.
            If `alternative_paths` is a list (or any other iterable) is
            provided, the file names will be replaced as they appear in `d`
            (note that this is very unsafe if `d` is not and OrderedDict)
        datasets: dict or None
            A mapping from original filenames in `d` to the instances of
            :class:`xarray.Dataset` to use.
        pwd: str
            Path to the working directory from where the data can be imported.
            If None, use the current working directory.
        ignore_keys: list of str
            Keys specified in this list are ignored and not seen as array
            information (note that ``attrs`` are used anyway)

        Other Parameters
        ----------------
        ``**kwargs``
            Any other parameter from the `psyplot.data.open_dataset` function
        %(open_dataset.parameters)s

        Returns
        -------
        psyplot.data.ArrayList
            The list with the interactive objects

        See Also
        --------
        from_dataset, array_info"""
        pwd = pwd or getcwd()
        if not isinstance(alternative_paths, dict):
            it = iter(alternative_paths)
            alternative_paths = defaultdict(partial(it.next, None))
        # first open all datasets if not already done
        if datasets is None:
            names_and_stores = cls._get_dsnames(deepcopy(d))
            datasets = {}
            for fname, (store_mod, store_cls) in names_and_stores:
                fname_use = fname
                got = True
                try:
                    fname_use = alternative_paths[fname]
                except KeyError:
                    got = False
                if not got or not fname_use:
                    if fname is not None:
                        if is_remote_url(fname):
                            fname_use = fname
                        else:
                            if os.path.isabs(fname):
                                fname_use = fname
                            else:
                                fname_use = os.path.join(pwd, fname)
                if fname_use is not None:
                    datasets[fname] = _open_ds_from_store(
                        fname_use, store_mod, store_cls, **kwargs)
            if alternative_paths is not None:
                for fname in set(alternative_paths).difference(datasets):
                    datasets[fname] = _open_ds_from_store(fname, **kwargs)
        arrays = [0] * len(set(d) - {'attrs'})
        for i, (arr_name, info) in enumerate(six.iteritems(d)):
            if arr_name in ignore_keys:
                continue
            if 'fname' not in info:
                arr = InteractiveList.from_dict(
                    info, alternative_paths=alternative_paths,
                    datasets=datasets)
            else:
                fname = info['fname']
                if fname is None:
                    warn("Could not open array %s because no filename was "
                         "specified!" % arr_name)
                    arrays.pop(i)
                    continue
                elif fname not in datasets:
                    warn("Could not open array %s because %s was not in the "
                         "list of datasets!")
                    arrays.pop(i)
                    continue
                arr = cls.from_dataset(
                    datasets[fname], dims=info['dims'], name=info['name'])[0]
                for key, val in six.iteritems(info.get('attrs', {})):
                    arr.attrs.setdefault(key, val)
            arr.arr_name = arr_name
            arrays[i] = arr
        return cls(arrays, attrs=d.get('attrs', {}))

    docstrings.delete_params('get_filename_ds.parameters', 'ds')

    @docstrings.get_sectionsf('ArrayList.array_info')
    @docstrings.dedent
    def array_info(self, dump=False, paths=None, attrs=True,
                   standardize_dims=True, pwd=None, use_rel_paths=True,
                   alternate_paths={}, ds_description={'fname', 'store'},
                   **kwargs):
        """
        Get dimension informations on you arrays

        This method returns a dictionary containing informations on the
        array in this instance

        Parameters
        ----------
        %(get_filename_ds.parameters.no_ds)s
        attrs: bool, optional
            If True (default), the :attr:`ArrayList.attrs` and
            :attr:`xarray.DataArray.attrs` attributes are included in the
            returning dictionary
        standardize_dims: bool, optional
            If True (default), the real dimension names in the dataset are
            replaced by x, y, z and t to be more general.
        pwd: str
            Path to the working directory from where the data can be imported.
            If None, use the current working directory.
        use_rel_paths: bool, optional
            If True (default), paths relative to the current working directory
            are used. Otherwise absolute paths to `pwd` are used
        ds_description: 'all' or set of {'fname', 'ds', 'num', 'arr', 'store'}
            Keys to describe the datasets of the arrays. If all, all keys
            are used. The key descriptions are

            fname
                the file name is inserted in the ``'fname'`` key
            store
                the data store class and module is inserted in the ``'store'``
                key
            ds
                the dataset is inserted in the ``'ds'`` key
            num
                The unique number assigned to the dataset is inserted in the
                ``'num'`` key
            arr
                The array itself is inserted in the ``'arr'`` key


        Other Parameters
        ----------------
        %(get_filename_ds.other_parameters)s

        Returns
        -------
        OrderedDict
            An ordered mapping from array names to dimensions and filename
            corresponding to the array

        See Also
        --------
        from_dict"""
        ret = OrderedDict()
        if ds_description == 'all':
            ds_description = {'fname', 'ds', 'num', 'arr', 'store'}
        if paths is not None:
            paths = iter(paths)
        if pwd is None:
            pwd = getcwd()
        for arr in self:
            if isinstance(arr, InteractiveList):
                ret[arr.arr_name] = arr.array_info(
                    dump, paths, pwd=pwd, attrs=attrs,
                    standardize_dims=standardize_dims,
                    use_rel_paths=use_rel_paths, ds_description=ds_description,
                    alternate_paths=alternate_paths, **kwargs)
            else:
                if standardize_dims:
                    idims = arr.decoder.standardize_dims(
                        next(arr.iter_base_variables), arr.idims)
                else:
                    idims = arr.idims
                ret[arr.arr_name] = d = {'dims': idims}
                if 'variable' in arr.coords:
                    d['name'] = [list(arr.coords['variable'].values)]
                else:
                    d['name'] = arr.name
                if 'fname' in ds_description or 'store' in ds_description:
                    fname, store_mod, store_cls = get_filename_ds(
                        arr.base, dump=dump, paths=paths, **kwargs)
                    if 'store' in ds_description:
                        d['store'] = (store_mod, store_cls)
                    if ('fname' in ds_description and
                            (fname is None or is_remote_url(fname))):
                        d['fname'] = fname
                    elif 'fname' in ds_description:
                        fname = alternate_paths.get(fname, alternate_paths.get(
                            os.path.abspath(fname), fname))
                        if fname in alternate_paths:
                            fname = alternate_paths[fname]
                        elif use_rel_paths:
                            fname = os.path.relpath(fname, pwd)
                        else:
                            fname = os.path.abspath(fname)
                        d['fname'] = fname
                if 'ds' in ds_description:
                    d['ds'] = arr.base
                if 'num' in ds_description:
                    d['num'] = self._get_psyplot_num(arr.base)
                if 'arr' in ds_description:
                    d['arr'] = arr
                if attrs:
                    d['attrs'] = arr.attrs
        ret['attrs'] = self.attrs
        return ret

    @staticmethod
    def _get_psyplot_num(ds, nums=None):
        """Assign a unique number to the dataset"""
        if nums is None:
            nums = _ds_counter
        if not hasattr(ds, PSYPLOT_ATTR):
            ds._initialized = False
            setattr(ds, PSYPLOT_ATTR, next(nums))
            ds._initialized = True
        return getattr(ds, PSYPLOT_ATTR)

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
    def start_update(self, draw=None):
        """
        Conduct the registered plot updates

        This method starts the updates from what has been registered by the
        :meth:`update` method. You can call this method if you did not set the
        `auto_update` parameter when calling the :meth:`update` method to True
        and when the :attr:`no_auto_update` attribute is True.

        Parameters
        ----------
        draw: bool or None
            If True, all the figures of the arrays contained in this list will
            be drawn at the end. If None, it defaults to the `'auto_draw'``
            parameter in the :attr:`psyplot.rcParams` dictionary

        See Also
        --------
        :attr:`no_auto_update`, update"""
        def worker(arr):
            results[arr.arr_name] = arr.start_update(draw=False, queues=queues)
        if len(self) == 0:
            return

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
        for thread in threads:
            thread.setDaemon(True)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if draw is None:
            draw = rcParams['auto_draw']
        if draw:
            self(arr_name=[name for name, adraw in six.iteritems(results)
                           if adraw]).draw()
            if rcParams['auto_show']:
                self.show()

    docstrings.keep_params('InteractiveArray.update.parameters',
                           'auto_update')

    @docstrings.get_sectionsf('ArrayList.update')
    @docstrings.dedent
    def update(self, method='isel', dims={}, fmt={}, replot=False,
               auto_update=False, draw=None, force=False, todefault=False,
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
        no_auto_update, start_update"""
        dims = dict(dims)
        fmt = dict(fmt)
        vars_and_coords = set(chain(
            self.dims, self.coords, ['name', 'x', 'y', 'z', 't']))
        furtherdims, furtherfmt = sort_kwargs(kwargs, vars_and_coords)
        dims.update(furtherdims)
        fmt.update(furtherfmt)

        self._register_update(method=method, replot=replot, dims=dims, fmt=fmt,
                              force=force, todefault=todefault)
        if not self.no_auto_update or auto_update:
            self.start_update(draw)

    def draw(self):
        """Draws all the figures in this instance"""
        for fig in set(chain(*map(
                lambda arr: arr.plotter.figs2draw, self.with_plotter))):
            self.logger.debug("Drawing figure %s", fig.number)
            fig.canvas.draw()
        for arr in self:
            if arr.plotter is not None:
                arr.plotter._figs2draw.clear()
        self.logger.debug("Done drawing.")

    def __call__(self, types=None, method='isel', arr_name=None, **attrs):
        """Get the arrays specified by their attributes

        Parameters
        ----------
        types: type or tuple of types
            Any class that shall be used for an instance check via
            :func:`isinstance`. If not None, the :attr:`plotter` attribute
            of the array is checked against this `types`
        method: {'isel', 'sel'}
            Selection method for the dimensions in the arrays to be used.
            If `method` is 'isel', dimension values in `attrs` must correspond
            to integer values as they are found in the
            :attr:`InteractiveArray.idims` attribute.
            Otherwise the :meth:`xarray.DataArray.coords` attribute is used.
        arr_name: None, str or list of str
            If not None, the array name serves as a filter
        ``**attrs``
            Parameters may be any attribute of the arrays in this instance.
            Values may be iterables (e.g. lists) of the attributes to consider.
            If the value is a string, it will be put into a list."""
        def safe_item_list(key, val):
            return key, safe_list(val)
        if not attrs:
            def filter_by_attrs(arr):
                return True
        elif method == 'sel':
            def filter_by_attrs(arr):
                if not isinstance(arr, InteractiveList):
                    tname = arr.decoder.get_tname(
                        next(six.itervalues(arr.base_variables)))

                    def check_values(attr, vals):
                        if hasattr(arr, 'decoder') and (
                                attr.name == tname):
                            try:
                                vals = np.asarray(vals, dtype=np.datetime64)
                            except ValueError:
                                pass
                            else:
                                return attr.values.astype(vals.dtype) in vals
                        return getattr(attr, 'values', attr) in vals

                return all(
                    check_values(getattr(arr, key, _NODATA), val)
                    for key, val in six.iteritems(
                        attrs if isinstance(arr, InteractiveList) else
                        arr.decoder.correct_dims(next(six.itervalues(
                            arr.base_variables)), attrs)))
        else:
            def filter_by_attrs(arr):
                if isinstance(arr, InteractiveList):
                    return all(
                        getattr(arr, key, _NODATA) in val
                        for key, val in six.iteritems(attrs))
                return all(
                    getattr(arr, key, _NODATA) if key not in arr.coords else (
                        arr.idims.get(key, _NODATA)) in val
                    for key, val in six.iteritems(
                        arr.decoder.correct_dims(next(six.itervalues(
                            arr.base_variables)), attrs)))
        if arr_name is not None and isstring(arr_name):
            arr_name = [arr_name]
        attrs = dict(starmap(safe_item_list, six.iteritems(attrs)))
        return self.__class__(
            # iterable
            (arr for arr in self if
             (types is None or isinstance(arr.plotter, types)) and
             (arr_name is None or arr.arr_name in arr_name) and
             filter_by_attrs(arr)),
            # give itself as base and the auto_update parameter
            auto_update=bool(self.auto_update))

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
        arr = self(arr_name=val.arr_name)[0]
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
            return all(a in arr for a in val) and all(a in val for a in arr)
        # else we check the shapes and values
        return arr is val

    def _short_info(self, intend=0, maybe=False):
        if maybe:
            intend = 0
        str_intend = ' ' * intend
        if len(self) == 1:
            return str_intend + "%s%s.%s([%s])" % (
                '' if not hasattr(self, 'arr_name') else self.arr_name + ': ',
                self.__class__.__module__, self.__class__.__name__,
                self[0]._short_info(intend+4, maybe=True))
        return str_intend + "%s%s.%s([\n%s])" % (
            '' if not hasattr(self, 'arr_name') else self.arr_name + ': ',
            self.__class__.__module__, self.__class__.__name__,
            ",\n".join(
                '%s' % (
                    arr._short_info(intend+4))
                for arr in self))

    def __str__(self):
        return self._short_info()

    def __repr__(self):
        return self.__str__()

    @docstrings.get_sectionsf('ArrayList.rename', sections=[
        'Parameters', 'Raises'])
    @dedent
    def rename(self, arr, new_name=True):
        """
        Rename an array to find a name that isn't already in the list

        Parameters
        ----------
        arr: InteractiveBase
            A :class:`InteractiveArray` or :class:`InteractiveList` instance
            whose name shall be checked
        new_name: bool or str
            If False, and the ``arr_name`` attribute of the new array is
            already in the list, a ValueError is raised.
            If True and the ``arr_name`` attribute of the new array is not
            already in the list, the name is not changed. Otherwise, if the
            array name is already in use, `new_name` is set to 'arr{0}'.
            If not True, this will be used for renaming (if the array name of
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
        if not name_in_me:
            return arr, False
        elif name_in_me and not self._contains_array(arr):
            if new_name is False:
                raise ValueError(
                    "Array name %s is already in use! Set the `new_name` "
                    "parameter to None for renaming!" % arr.arr_name)
            elif new_name is True:
                new_name = new_name if isstring(new_name) else 'arr{0}'
                arr.arr_name = self.next_available_name(new_name)
                return arr, True
        return arr, None

    def next_available_name(self, fmt_str='arr{0}', counter=None):
        """Create a new array out of the given format string

        Parameters
        ----------
        format_str: str
            The base string to use. ``'{0}'`` will be replaced by a counter
        counter: iterable
            An iterable where the numbers should be drawn from. If None,
            ``range(100)`` is used

        Returns
        -------
        str
            A possible name that is not in the current project"""
        names = self.arr_names
        counter = counter or iter(range(1000))
        try:
            new_name = next(
                filter(lambda n: n not in names,
                       map(fmt_str.format, counter)))
        except StopIteration:
            raise ValueError(
                "{0} already in the list".format(fmt_str))
        return new_name

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
    def extend(self, iterable, new_name=False):
        """
        Add further arrays from an iterable to this list

        Parameters
        ----------
        iterable
            Any iterable that contains :class:`InteractiveBase` instances
        %(ArrayList.rename.parameters.new_name)s

        Raises
        ------
        %(ArrayList.rename.raises)s

        See Also
        --------
        list.extend, append, rename"""
        # extend those arrays that aren't alredy in the list
        super(ArrayList, self).extend(t[0] for t in filter(
            lambda t: t[1] is not None, (
                self.rename(arr, new_name) for arr in iterable)))

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
        if arr not in self:
            raise ValueError(
                "Array {0} not in the list".format(name))
        for i, arr in enumerate(self):
            if arr.arr_name == name:
                del self[i]
                return
        raise ValueError(
            "No array found with name {0}".format(name))


class InteractiveList(ArrayList, InteractiveBase):
    """List of :class:`InteractiveArray` instances that can be plotted itself

    This class combines the :class:`ArrayList` and the interactive plotting
    through :class:`psyplot.plotter.Plotter` classes. It is mainly used by the
    :mod:`psyplot.plotter.simple` module"""

    no_auto_update = property(_no_auto_update_getter,
                              doc=_no_auto_update_getter.__doc__)

    @no_auto_update.setter
    def no_auto_update(self, value):
        ArrayList.no_auto_update.fset(self, value)
        InteractiveBase.no_auto_update.fset(self, value)

    @property
    @docstrings
    def _njobs(self):
        """%(InteractiveBase._njobs)s"""
        ret = super(self.__class__, self)._njobs or [0]
        ret[0] += 1
        return ret

    logger = InteractiveBase.logger

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
    def start_update(self, draw=None, queues=None):
        """
        Conduct the formerly registered updates

        This method conducts the updates that have been registered via the
        :meth:`update` method. You can call this method if the
        :attr:`auto_update` attribute of this instance is True and the
        `auto_update` parameter in the :meth:`update` method has been set to
        False

        Parameters
        ----------
        %(InteractiveBase.start_update.parameters)s

        Returns
        -------
        %(InteractiveBase.start_update.returns)s

        See Also
        --------
        :attr:`no_auto_update`, update
        """
        if queues is not None:
            queues[0].get()
        try:
            for arr in self:
                arr.start_update(draw=False)
            self.onupdate.emit()
        except:
            self._finish_all(queues)
            raise
        if queues is not None:
            queues[0].task_done()
        return InteractiveBase.start_update(self, draw=draw, queues=queues)

    def to_dataframe(self):
        def to_df(arr):
            df = arr.to_pandas()
            if hasattr(df, 'to_frame'):
                df = df.to_frame()
            if not keep_names:
                return df.rename(columns={df.keys()[0]: arr.arr_name})
            return df
        if len(self) == 1:
            return self[0].to_series().to_frame()
        else:
            keep_names = len(set(arr.name for arr in self)) == self
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


def _open_ds_from_store(fname, store_mod=None, store_cls=None, **kwargs):
    """Open a dataset and return it"""
    if isinstance(fname, xarray.Dataset):
        return fname
    if store_mod is not None and store_cls is not None:
        fname = getattr(import_module(store_mod), store_cls)(fname)
    return open_dataset(fname, **kwargs)


def decode_absolute_time(times):
    def decode(t):
        day, sub = re.findall('(\d+)(\.\d+)', t)[0]
        return np.datetime64(
            datetime.strptime(day, "%Y%m%d") + timedelta(days=float(sub)))
    times = np.asarray(times, dtype=str)
    return np.vectorize(decode, [np.datetime64])(times)


def encode_absolute_time(times):
    def encode(t):
        t = to_datetime(t)
        return float(t.strftime('%Y%m%d')) + (
            t - datetime(t.year, t.month, t.day)).total_seconds() / 86400.
    return np.vectorize(encode, [float])(times)


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
            self._dtype = getattr(result, 'dtype', np.dtype('object'))

    @property
    def dtype(self):
        return self._dtype

    def __getitem__(self, key):
        return decode_absolute_time(self.array[key])


class AbsoluteTimeEncoder(NDArrayMixin):

    def __init__(self, array):
        self.array = array
        example_value = first_n_items(array, 1) or 0
        try:
            result = encode_absolute_time(example_value)
        except Exception:
            logger.error("Could not interprete absolute time values!")
            raise
        else:
            self._dtype = getattr(result, 'dtype', np.dtype('object'))

    @property
    def dtype(self):
        return self._dtype

    def __getitem__(self, key):
        return encode_absolute_time(self.array[key])
