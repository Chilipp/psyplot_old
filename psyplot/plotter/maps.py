import six
import re
import yaml
from abc import abstractproperty
from difflib import get_close_matches
from itertools import starmap, chain, repeat
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import Gridliner
import matplotlib as mpl
import matplotlib.ticker as ticker
import numpy as np
from .. import rcParams
from ..compat.pycompat import map, range
from ..docstring import docstrings
from ..warning import warn, PsyPlotRuntimeWarning
from ..data import InteractiveList
from . import Formatoption, START, DictFormatoption, END
from .simpleplotter import (
    Base2D, Plot2D, BasePlotter, BaseVectorPlotter, VectorPlot, CombinedBase,
    _infer_interval_breaks, DataTicksCalculator, round_to_05, Density,
    Simple2DBase, DataGrid, VectorColor, get_cmap)
from .boxes import lonlatboxes


@docstrings.get_sectionsf('shiftdata')
def shiftdata(lonsin, datain, lon_0):
    """
    Shift longitudes (and optionally data) so that they match map projection
    region.
    Only valid for cylindrical/pseudo-cylindrical global projections and data
    on regular lat/lon grids. longitudes and data can be 1-d or 2-d, if 2-d
    it is assumed longitudes are 2nd (rightmost) dimension.

    Parameters
    ----------
    lonsin
        original 1-d or 2-d longitudes.
    datain
        original 1-d or 2-d data
    lon_0
        center of map projection region

    References
    ----------
    This function is copied and taken from the
    :class:`mpl_toolkits.basemap.Basemap` class. The only difference is that
    we do not mask values outside the map projection region
    """
    lonsin = np.asarray(lonsin)
    if lonsin.ndim not in [1, 2]:
        raise ValueError('1-d or 2-d longitudes required')
    if datain is not None:
        # if it's a masked array, leave it alone.
        if not np.ma.isMA(datain):
            datain = np.asarray(datain)
        if datain.ndim not in [1, 2]:
            raise ValueError('1-d or 2-d data required')
    # 2-d data.
    if lonsin.ndim == 2:
        raise NotImplementedError(
            "Shifting of 2D-data is currently not supported")
        nlons = lonsin.shape[1]
        lonsin1 = lonsin[0, :]
        lonsin1 = np.where(lonsin1 > lon_0+180, lonsin1-360, lonsin1)
        lonsin1 = np.where(lonsin1 < lon_0-180, lonsin1+360, lonsin1)
        londiff = np.abs(lonsin1[0:-1]-lonsin1[1:])
        londiff_sort = np.sort(londiff)
        thresh = 360.-londiff_sort[-2]
        itemindex = nlons - np.where(londiff >= thresh)[0]
        # if no shift necessary, itemindex will be
        # empty, so don't do anything
        if len(itemindex):
            # check to see if cyclic (wraparound) point included
            # if so, remove it.
            if np.abs(lonsin1[0]-lonsin1[-1]) < 1.e-4:
                hascyclic = True
                lonsin_save = lonsin.copy()
                lonsin = lonsin[:, 1:]
                if datain is not None:
                    datain_save = datain.copy()
                    datain = datain[:, 1:]
            else:
                hascyclic = False
            lonsin = np.where(lonsin > lon_0+180, lonsin-360, lonsin)
            lonsin = np.where(lonsin < lon_0-180, lonsin+360, lonsin)
            lonsin = np.roll(lonsin, itemindex-1, axis=1)
            if datain is not None:
                # np.roll works on ndarrays and on masked arrays
                datain = np.roll(datain, itemindex-1, axis=1)
            # add cyclic point back at beginning.
            if hascyclic:
                lonsin_save[:, 1:] = lonsin
                lonsin_save[:, 0] = lonsin[:, -1] - 360.
                lonsin = lonsin_save
                if datain is not None:
                    datain_save[:, 1:] = datain
                    datain_save[:, 0] = datain[:, -1]
                    datain = datain_save
    # 1-d data.
    elif lonsin.ndim == 1:
        nlons = len(lonsin)
        lonsin = np.where(lonsin > lon_0+180, lonsin-360, lonsin)
        lonsin = np.where(lonsin < lon_0-180, lonsin+360, lonsin)
        londiff = np.abs(lonsin[0:-1]-lonsin[1:])
        londiff_sort = np.sort(londiff)
        thresh = 360.-londiff_sort[-2]
        itemindex = len(lonsin) - np.where(londiff >= thresh)[0]
        if itemindex:
            # check to see if cyclic (wraparound) point included
            # if so, remove it.
            if np.abs(lonsin[0]-lonsin[-1]) < 1.e-4:
                hascyclic = True
                lonsin_save = lonsin.copy()
                lonsin = lonsin[1:]
                if datain is not None:
                    datain_save = datain.copy()
                    datain = datain[1:]
            else:
                hascyclic = False
            lonsin = np.roll(lonsin, itemindex-1)
            if datain is not None:
                datain = np.roll(datain, itemindex-1)
            # add cyclic point back at beginning.
            if hascyclic:
                lonsin_save[1:] = lonsin
                lonsin_save[0] = lonsin[-1]-360.
                lonsin = lonsin_save
                if datain is not None:
                    datain_save[1:] = datain
                    datain_save[0] = datain[-1]
                    datain = datain_save
    if datain is not None:
        return lonsin, datain
    else:
        return lonsin


def degree_format():
    if mpl.rcParams['text.usetex']:
        return r'${%g\/^{\circ}\/%s}$'
    else:
        return u'%g\N{DEGREE SIGN}%s'


def format_lons(x, pos):
    fmt_string = degree_format()
    x = x - 360 if x > 180 else x
    if x == 0:
        return fmt_string % (x, '')
    return fmt_string % (abs(x), 'W' if x < 0 else 'E')


def format_lats(x, pos):
    fmt_string = degree_format()
    if x == 0:
        return fmt_string % (abs(x), '')
    return fmt_string % (abs(x), 'S' if x < 0 else 'N')


lon_formatter = ticker.FuncFormatter(format_lons)
lat_formatter = ticker.FuncFormatter(format_lats)


@docstrings.get_sectionsf('ProjectionBase')
class ProjectionBase(Formatoption):
    """
    Base class for formatoptions that uses cartopy.crs.CRS instances

    Possible types
    --------------
    cartopy.crs.CRS
        A cartopy projection instance (e.g. :class:`cartopy.crs.PlateCarree`)
    str
        A string specifies the projection instance to use. The centered
        longitude and latitude are determined by the :attr:`clon` and
        :attr:`clat` formatoptions.
        Possible strings are (each standing for the specified projection)

        =========== =======================================
        cyl         :class:`cartopy.crs.PlateCarree`
        robin       :class:`cartopy.crs.Robinson`
        moll        :class:`cartopy.crs.Mollweide`
        geo         :class:`cartopy.crs.Geostationary`
        northpole   :class:`cartopy.crs.NorthPolarStereo`
        southpole   :class:`cartopy.crs.SouthPolarStereo`
        ortho       :class:`cartopy.crs.Orthographic`
        =========== ======================================="""

    projections = {
        'cyl': ccrs.PlateCarree,
        'robin': ccrs.Robinson,
        'moll': ccrs.Mollweide,
        'geo': ccrs.Geostationary,
        'northpole': ccrs.NorthPolarStereo,
        'southpole': ccrs.SouthPolarStereo,
        'ortho': ccrs.Orthographic,
        }

    projection_kwargs = dict(
        chain(zip(projections.keys(), repeat(['central_longitude']))))
    projection_kwargs['ortho'] = ['central_longitude', 'central_latitude']

    def set_projection(self, value, *args, **kwargs):
        if isinstance(value, ccrs.CRS):
            return value
        else:
            return self.projections[value](**self.get_kwargs(
                value, *args, **kwargs))

    def get_kwargs(self, value, clon=None, clat=None):
        ret = {}
        keys = self.projection_kwargs[value]
        if 'central_longitude' in keys:
            ret['central_longitude'] = self.clon.clon if clon is None else clon
        if 'central_latitude' in keys:
            ret['central_latitude'] = self.clat.clat if clat is None else clat
        self.logger.debug("Setting projection with %s", ret)
        return ret


class Projection(ProjectionBase):
    """
    Specify the projection for the plot

    This formatoption defines the projection of the plot

    Possible types
    --------------
    %(ProjectionBase.possible_types)s

    Warnings
    --------
    An update of the projection clears the axes!
    """

    # the axes has to be cleared completly if the projection is updated
    priority = START

    requires_clearing = True

    dependencies = ['clon', 'clat']

    def __init__(self, *args, **kwargs):
        super(Projection, self).__init__(*args, **kwargs)
        self.projection = None

    def initialize_plot(self, value, clear=True):
        self.projection = self.set_projection(value)
        if self.plotter.cleared:
            self.ax.projection = self.projection
            self.ax.clear()

    def update(self, value):
        pass


@docstrings.get_sectionsf('BoxBase')
class BoxBase(Formatoption):
    """
    Abstract base class for specifying a longitude-latitude box

    Possible types
    --------------
    str
        A pattern that matches any of the keys in the :attr:`psyplot.rcParams`
        ``'extents.boxes'`` item (contains user-defined longitude-latitude
        boxes) or the :attr:`psyplot.plotter.boxes.lonlatboxes` dictionary
        (contains longitude-latitude boxes of different countries and
        continents)
    [lonmin, lonmax, latmin, latmax]
        The surrounding longitude-latitude that shall be used

    See Also
    --------
    LonLatBox, MapExtent
    """

    def lola_from_pattern(self, s):
        """
        Calculate the longitude-latitude box based upon a pattern

        This method uses the psyplot.rcParams ``'extents.boxes'`` item to find
        longitude that match `s` and takes the surrounding box.

        Parameters
        ----------
        s: str
            The pattern to use for the keys in the
            :attr:`psyplot.plotter.maps.lonlatboxes` dictionary and the
            ``'extents.boxes'`` item in the :attr:`psyplot.rcParams`

        Returns
        -------
        float: lonmin, lonmax, latmin, latmax or None
            The surrounding longitude-latitude box of all items in
            ``psyplot.rcParams['extents.boxes']`` whose key match `s` if there
            was any match. Otherwise None is returned
        """
        patt = re.compile(s)
        boxes = np.array([
            box for key, box in chain(*map(
                six.iteritems, [lonlatboxes, rcParams['lonlatbox.boxes']]))
            if patt.search(key)])
        if len(boxes) == 0:
            similar_keys = get_close_matches(s, rcParams['lonlatbox.boxes'])
            message = "Did not find any matches for %s!" % s
            if similar_keys:
                message += " Maybe you mean on of " + ', '.join(
                    similar_keys)
            warn(message, PsyPlotRuntimeWarning,
                 logger=self.logger)
            return
        return [boxes[:, 0].min(), boxes[:, 1].max(),
                boxes[:, 2].min(), boxes[:, 3].max()]


docstrings.keep_types('BoxBase.possible_types', 'str', 'str')


class CenterLon(BoxBase):
    """
    Set the center longitude of the plot

    Parameters
    ----------
    None
        Let the :attr:`lonlatbox` formatoption determine the center
    float
        Specifiy the center manually
    %(BoxBase.possible_types.str)s
    """

    priority = START

    requires_clearing = True

    dependencies = ['lonlatbox']

    def update(self, value):
        self.lon_mean = np.mean(self.lonlatbox.lonlatbox[:2])
        if value is not None:
            if isinstance(value, six.string_types):
                box = self.lola_from_pattern(value)
                if box is not None:
                    self.clon = np.mean(box[:2])
                else:
                    value = None
            else:
                self.clon = value
        if value is None and self.lonlatbox.value is not None:
            self.clon = self.lon_mean
        elif value is None:
            self.clon = 0.0


class CenterLat(BoxBase):
    """
    Set the center latitude of the plot

    Parameters
    ----------
    None
        Let the :attr:`lonlatbox` formatoption determine the center
    float
        Specifiy the center manually
    %(BoxBase.possible_types.str)s
    """

    priority = START

    requires_clearing = True

    dependencies = ['lonlatbox']

    def update(self, value):
        self.lat_mean = np.mean(self.lonlatbox.lonlatbox[2:])
        if value is not None:
            if isinstance(value, six.string_types):
                box = self.lola_from_pattern(value)
                if box is not None:
                    self.clat = np.mean(box[2:])
                else:
                    value = None
            else:
                self.clat = value
        if value is None and self.lonlatbox.value is not None:
            self.clat = self.lat_mean
        elif value is None:
            self.clat = 0.0


@docstrings.get_sectionsf('LonLatBox')
class LonLatBox(BoxBase):
    """
    Set the longitude-latitude box of the data shown

    This formatoption extracts the data that matches the specified box.

    Possible types
    --------------
    None
        Use the full data
    %(BoxBase.possible_types)s

    Notes
    -----
    - For only specifying the region of the plot, see the :attr:`map_extent`
      formatoption
    - If the coordinates are two-dimensional (e.g. for a circumpolar grid),
      than the data is not extracted but values outside the specified
      longitude-latitude box are set to NaN

    See Also
    --------
    map_extent"""

    priority = START

    requires_clearing = True

    dependencies = ['transform']

    def data_dependent(self, data, set_data=True):
        if isinstance(data, InteractiveList):
            data = data[0]
        decoder = data.decoder
        lon, lat = self._get_lola(data, decoder)
        new_lonlatbox = self.calc_lonlatbox(lon, lat)
        update = self.data_lonlatbox != new_lonlatbox
        if not update and set_data and self.value is not None:
            self.update(self.value)
        elif update:
            self.logger.debug(
                "Reinitializing because lonlatbox of new data %s does not "
                "match the old one %s", new_lonlatbox, self.data_lonlatbox)
        return update

    def update(self, value):
        if isinstance(self.data, InteractiveList):
            for i, arr in enumerate(self.data):
                decoder = self.raw_data[i].decoder
                self.data[i] = self.update_array(
                    value, arr, decoder, next(six.itervalues(
                        self.raw_data[i].base_variables)))
        else:
            arr = self.data
            arr = self.update_array(
                    value, arr, self.raw_data.decoder, next(six.itervalues(
                        self.raw_data.base_variables)))
            self.data = arr

    def _get_lola(self, data, decoder):
        """Get longitude and latitde informations from the given data array"""
        lon_da = decoder.get_x(data, data.coords)
        lat_da = decoder.get_y(data, data.coords)
        lon, lat = self.to_degree(lon_da.attrs.get('units'), lon_da.values,
                                  lat_da.values)
        data_shape = data.shape[-2:] if not decoder.is_unstructured(data) \
            else (data.shape[-1], )
        if lon.shape == data_shape and lat.shape == data_shape:
            i = len(data_shape)
            if i < len(data.shape):
                comp_data = np.all(~np.isnan(data.values), axis=0)
                for j in range(1, len(data.shape) - i):
                    comp_data = np.all(~np.isnan(comp_data), axis=(j - 1))
            else:
                comp_data = ~np.isnan(data.values)
            lon = lon[comp_data]
            lat = lat[comp_data]
        return lon, lat

    def update_array(self, value, data, decoder, base_var=None):
        """Update the given `data` array"""
        lon, lat = self._get_lola(data, decoder)
        self.data_lonlatbox = self.calc_lonlatbox(lon, lat)
        if value is None:
            self.lonlatbox = self.data_lonlatbox
            return data
        else:
            if isinstance(value, six.string_types):
                value = self.lola_from_pattern(value)
                if value is None:
                    self.lonlatbox = self.data_lonlatbox
                    return data
            lat_values = value[3:1:-1] if (lat[1:] < lat[:-1]).all() else \
                value[2:]
            is_unstructured = decoder.is_unstructured(
                base_var if base_var is not None else data)
            is_rectilinear = lon.ndim == 1 and not is_unstructured
            shift = isinstance(self.transform.projection, ccrs.PlateCarree)
            if is_rectilinear and shift:
                data = data.copy(True)
                lon_da, data.values = self.shiftdata(
                    lon, data.values, np.mean(value[:2]))
                data.lon.values = lon_da
            elif is_unstructured and shift:
                # make sure that we are inside the map extent
                ret = self.transform.projection.transform_points(
                    self.transform.projection, lon, lat)
                lon = ret[..., 0]
                lat = ret[..., 1]
            self.lonlatbox = value
            if is_rectilinear:
                kwargs = dict(zip(
                    data.dims[-2:], starmap(slice, [lat_values, value[:2]])))
                return data.sel(**kwargs)
            else:
                # set values outsite the lonlatbox to nan
                # TODO: that does currently not work!
                return self.mask_outside(data.copy(True), lon, lat, *value,
                                         is_unstructured=is_unstructured)

    def to_degree(self, units=None, *args):
        """Converts arrays with radian units to degree

        Parameters
        ----------
        units: str
            if ``'radian'``, the arrays in ``*args`` will be converted
        ``*args``
            numpy arrays

        Returns
        -------
        list of np.ndarray
            returns the arrays provided with ``*args``

        Notes
        -----
        if `units` is ``'radian'``, a copy of the array will be returned"""
        args = list(args)
        if units == 'radian' and isinstance(self.transform.projection,
                                            ccrs.PlateCarree):
            for i, array in enumerate(args):
                args[i] = array * 180. / np.pi
        return args

    def mask_outside(self, data, lon, lat, lonmin, lonmax, latmin, latmax,
                     is_unstructured=False):
        ndim = 2 if not is_unstructured else 1
        data.values = data.values.copy()
        for arr in data.values if data.ndim > ndim else [data.values]:
            arr[np.any([lon < lonmin, lon > lonmax, lat < latmin,
                        lat > latmax], axis=0)] = np.nan
        return data

    def calc_lonlatbox(self, lon, lat):
        if isinstance(self.transform.projection, ccrs.PlateCarree):
            lon = lon[np.all([lon >= -180, lon <= 360], axis=0)]
            lat = lat[np.all([lat >= -90, lat <= 90], axis=0)]
        return [lon.min(), lon.max(), lat.min(), lat.max()]

    def shiftdata(self, lonsin, datain, lon_0):
        """
        Shift the data such that it matches the region we want to show

        Parameters
        ----------
        %(shiftdata.parameters)s

        Notes
        -----
        `datain` can also be multiple fields stored in a three-dimensional
        array. Then we shift all fields along the first dimension
        """
        # shiftdata does not work properly if we do not give the bounds of
        # the array
        if lonsin.ndim == 1:
            get_centers = True
            lonsin = _infer_interval_breaks(lonsin)
        else:
            get_centers = False
        if datain.ndim == 2:
            lonsin, datain = shiftdata(lonsin, datain, lon_0)
        elif datain.ndim == 3:
            lon_save = lonsin.copy()
            for i, a in enumerate(datain):
                lonsin, datain[i] = shiftdata(lon_save, a, lon_0)
        if get_centers:
            lonsin = np.mean([lonsin[1:], lonsin[:-1]], axis=0)
        return lonsin, datain


class MapExtent(BoxBase):
    """
    Set the extent of the map

    Possible types
    --------------
    None
        The map extent is specified by the data (i.e. by the :attr:`lonlatbox`
        formatoption)
    'global'
        The whole globe is shown
    %(BoxBase.possible_types)s

    Notes
    -----
    This formatoption sets the extent of the plot. For choosing the region for
    the data, see the :attr:`lonlatbox` formatoption

    See Also
    --------
    lonlatbox"""

    dependencies = ['lonlatbox', 'plot', 'vplot']

    priority = END

    update_after_plot = True

    def update(self, value):
        set_global = False
        if isinstance(value, six.string_types):
            if value == 'global':
                set_global = True
            else:
                value = self.lola_from_pattern(value)
        elif value is None:
            value = self.lonlatbox.lonlatbox
        # Since the set_extent method does not always work and the data limits
        # are not always correctly set, we test here whether the wished
        # extent (the value) is almost global. If so, we set it to a global
        # value
        if set_global or ((value[1] - value[0]) / 360. > 0.95 or
                          (value[3] - value[2]) / 180. > 0.95):
            self.logger.debug("Setting to global extent...")
            self.ax.set_global()
        else:
            try:
                self.ax.set_extent(value, crs=ccrs.PlateCarree())
            except ValueError:
                self.logger.debug(
                    "Failed to set_extent with lonlatbox %s", value,
                    exc_info=True)


class Transform(ProjectionBase):
    """
    Specify the coordinate system of the data

    This formatoption defines the coordinate system of the data (usually we
    expect a simple latitude longitude coordinate system)

    Possible types
    --------------
    %(ProjectionBase.possible_types)s
    """

    priority = START

    connections = ['plot', 'vplot']

    def update(self, value):
        self.projection = self.set_projection(value, 0, 0)
        for key in self.connections:
            try:
                getattr(self, key)._kwargs['transform'] = self.projection
            except AttributeError:
                pass


class LSM(Formatoption):
    """
    Draw the continents

    Possible types
    --------------
    bool
        True: draw the continents with a line width of 1
        False: don't draw the continents
    float
        Specifies the linewidth of the continents"""

    def update(self, value):
        if value:
            value = 1.0 if value is True else value
            self.lsm = self.ax.coastlines(linewidth=value)
        elif hasattr(self, 'lsm'):
            self.lsm.remove()
            del self.lsm


class GridColor(Formatoption):
    """
    Set the color of the grid

    Possible types
    --------------
    None
        Choose the default line color
    color
        Any valid color for matplotlib (see the :func:`matplotlib.pyplot.plot`
        documentation)

    See Also
    --------
    grid_settings, grid_labels, grid_labelsize, xgrid, ygrid"""

    connections = ['xgrid', 'ygrid']

    def update(self, value):
        if value is not None:
            for connection in self.connections:
                getattr(self, connection)._kwargs['color'] = value
        else:
            for connection in self.connections:
                getattr(self, connection)._kwargs.pop('color', None)


class GridLabels(Formatoption):
    """
    Display the labels of the grid

    Possible types
    --------------
    bool
        If True, labels are drawn

    See Also
    --------
    grid_color, grid_settings, grid_labelsize, xgrid, ygrid"""

    dependencies = ['projection', 'transform']

    connections = ['xgrid', 'ygrid']

    def update(self, value):
        try:  # initialize a gridliner to see if we can draw the tick labels
            Gridliner(self.ax, self.transform.projection, draw_labels=value)
        except TypeError as e:  # labels cannot be drawn
            value = False
            warn(e.message, PsyPlotRuntimeWarning, logger=self.logger)
        for connection in self.connections:
            getattr(self, connection)._kwargs['draw_labels'] = value


class GridSettings(DictFormatoption):
    """
    Modify the settings of the grid explicitly

    Possible types
    --------------
    dict
        Items may be any key-value-pair of the
        :class:`matplotlib.collections.LineCollection` class

    See Also
    --------
    grid_color, grid_labels, grid_labelsize, xgrid, ygrid"""

    children = ['grid_labels', 'grid_color']
    connections = ['xgrid', 'ygrid']

    def set_value(self, value, validate=True, todefault=False):
        if todefault:
            for connection in self.connections:
                for key in self.value:
                    getattr(self, connection)._kwargs.pop(key, None)
        super(GridSettings, self).set_value(value, validate, todefault)

    def update(self, value):
        for connection in self.connections:
            getattr(self, connection)._kwargs.update(value)


class GridLabelSize(Formatoption):
    """
    Modify the size of the grid tick labels

    Possible types
    --------------
    %(fontsizes)s

    See Also
    --------
    grid_color, grid_labels, xgrid, ygrid, grid_settings"""

    dependencies = ['xgrid', 'ygrid']

    def update(self, value):
        for fmto in map(lambda key: getattr(self, key), self.dependencies):
            try:
                gl = fmto._gridliner
            except AttributeError:
                continue
            if self.plotter._initializing or self.plotter.has_changed(
                    fmto.key):
                gl.xlabel_style['size'] = value
                gl.ylabel_style['size'] = value
            else:
                for text in chain(gl.xlabel_artists, gl.ylabel_artists):
                    text.set_size(value)


@docstrings.get_sectionsf('GridBase', sections=['Possible types', 'See Also'])
class GridBase(DataTicksCalculator):
    """
    Abstract base class for x- and y- grid lines

    Possible types
    --------------
    None
        Don't draw gridlines (same as ``False``)
    bool
        True: draw gridlines and determine position automatically
        False: don't draw gridlines
    %(DataTicksCalculator.possible_types)s
    int
        Specifies how many ticks to use with the ``'rounded'`` option. I.e. if
        integer ``i``, then this is the same as ``['rounded', i]``.

    See Also
    --------
    grid_color, grid_labels"""

    dependencies = ['transform', 'grid_labels', 'grid_color', 'grid_settings',
                    'projection', 'lonlatbox', 'map_extent']

    @abstractproperty
    def axis(self):
        """The axis string"""
        pass

    def __init__(self, *args, **kwargs):
        super(GridBase, self).__init__(*args, **kwargs)
        self._kwargs = {}

    def update(self, value):
        self.remove()
        if value is None or value is False:
            return
        if value is True:
            loc = None
        elif isinstance(value[0], six.string_types):
            loc = ticker.FixedLocator(self.calc_funcs[value[0]](*value[1:]))
        elif isinstance(value, tuple):
            steps = 11 if len(value) == 2 else value[3]
            loc = ticker.FixedLocator(
                np.linspace(value[0], value[1], steps, endpoint=True))
        else:
            loc = ticker.FixedLocator(value)
        self._gridliner = self.ax.gridlines(
            self.transform.projection, **self.get_kwargs(loc))
        self._modify_gridliner(self._gridliner)
        self._disable_other_axis()

    def get_kwargs(self, loc):
        return dict(chain(self._kwargs.items(), [(self.axis + 'locs', loc)]))

    def _disable_other_axis(self):
        label_positions = {'x': ['bottom', 'top'], 'y': ['left', 'right']}
        other_axis = 'y' if self.axis == 'x' else 'x'
        setattr(self._gridliner, other_axis + 'lines', False)
        for pos in label_positions[other_axis]:
            setattr(self._gridliner, other_axis + 'labels_' + pos, False)

    def _modify_gridliner(self, gridliner):
        """Modify the formatting of the given `gridliner` before drawing"""
        gridliner.xlabels_top = False
        gridliner.ylabels_right = False
        gridliner.yformatter = lat_formatter
        gridliner.xformatter = lon_formatter

    def remove(self):
        if not hasattr(self, '_gridliner'):
            return
        gl = self._gridliner
        for artist in chain(gl.xline_artists, gl.yline_artists,
                            gl.xlabel_artists, gl.ylabel_artists):
            artist.remove()
        if gl in self.ax._gridliners:
            self.ax._gridliners.remove(gl)
        del self._gridliner

    def _round_min_max(self, vmin, vmax):
        exp = np.floor(np.log10(abs(vmax - vmin)))
        return round_to_05([vmin, vmax], exp, mode='s')


class XGrid(GridBase):
    """
    Draw vertical grid lines (meridians)

    This formatoption specifies at which longitudes to draw the meridians.

    Possible types
    --------------
    %(GridBase.possible_types)s

    See Also
    --------
    ygrid, %(GridBase.see_also)s"""

    dependencies = GridBase.dependencies + ['clon']

    @property
    def array(self):
        decoder = self.raw_data.decoder
        coord = decoder.get_x(self.data, self.data.coords)
        arr = np.unique(decoder.get_plotbounds(coord, ignore_shape=True))
        if hasattr(coord, 'units') and coord.units == 'radian':
            arr *= 180. / np.pi
        if isinstance(self.ax.projection, ccrs.PlateCarree) and isinstance(
                self.transform.projection, ccrs.PlateCarree):
            arr = self.ax.projection.transform_points(
                self.transform.projection, arr, np.zeros(arr.shape))[..., 0]
        return arr

    axis = 'x'


class YGrid(GridBase):
    """
    Draw horizontal grid lines (parallels)

    This formatoption specifies at which latitudes to draw the parallels.

    Possible types
    --------------
    %(GridBase.possible_types)s

    See Also
    --------
    xgrid, %(GridBase.see_also)s"""

    @property
    def array(self):
        decoder = self.raw_data.decoder
        coord = decoder.get_y(self.data, self.data.coords)
        arr = np.unique(decoder.get_plotbounds(coord, ignore_shape=True))
        if hasattr(coord, 'units') and coord.units == 'radian':
            arr *= 180. / np.pi
        return arr

    axis = 'y'


class MapPlot2D(Plot2D):
    __doc__ = Plot2D.__doc__
    # fixes the plot of unstructured triangular data on round projections

    connections = Plot2D.connections + ['transform']

    def _tripcolor(self):
        from matplotlib.tri import TriAnalyzer
        mratio = rcParams['plotter.maps.plot.min_circle_ratio']
        triangles = self.triangles
        if mratio:
            flat_mask = TriAnalyzer(triangles).get_flat_tri_mask(mratio)
            # we have to apply a little workaround here in order to draw the
            # boundaries right. That implies that we mask out flat triangles
            # (epecially those at the end) and transform them manually
            decoder = self.raw_data.decoder
            triangles_wrapped = decoder.get_triangles(
                self.data, self.data.coords, copy=True,
                src_crs=self.transform.projection,
                target_crs=self.ax.projection)
            triangles_wrapped.set_mask(~flat_mask)
            triangles.set_mask(flat_mask)
        cmap = get_cmap(self.cmap.value, len(self.bounds.bounds) - 1 or None)
        if self.miss_color and self.miss_color.value is not None:
            # does not work for tripcolor (for whatever reason)
            warn('The miss_color formatoption does not work for triangular '
                 'plots', PsyPlotRuntimeWarning, logger=self.logger)
#            cmap.set_bad(self.miss_color.value)
        if hasattr(self, '_plot'):
            self._plot.update(dict(cmap=cmap, norm=self.bounds.norm))
        else:
            self._plot = self.ax.tripcolor(
                triangles, self.array, norm=self.bounds.norm, cmap=cmap,
                rasterized=True, **self._kwargs)
        # draw wrapped collection to fix the issue that the boundaries are
        # masked out when using the min_circle_ration
        if mratio and not hasattr(self, '_wrapped_plot'):
            kwargs = self._kwargs.copy()
            kwargs['zorder'] = self._plot.zorder - 0.1
            kwargs.pop('transform', None)
            kwargs.setdefault('snap', False)
            self._wrapped_plot = self.ax.tripcolor(
                triangles_wrapped, self.array, norm=self.bounds.norm,
                cmap=cmap, rasterized=True, **kwargs)
        else:
            self._wrapped_plot.update(dict(cmap=cmap, norm=self.bounds.norm))
        return

    def remove(self):
        super(MapPlot2D, self).remove()
        if hasattr(self, '_wrapped_plot'):
            self._wrapped_plot.remove()
            del self._wrapped_plot


class MapDataGrid(DataGrid):
    __doc__ = DataGrid.__doc__

    def triangles(self):
        from matplotlib.tri import TriAnalyzer
        decoder = self.raw_data.decoder
        triangles = decoder.get_triangles(
            self.data, self.data.coords, copy=True,
            src_crs=self.transform.projection, target_crs=self.ax.projection)
        mratio = rcParams['plotter.maps.plot.min_circle_ratio']
        if mratio:
            triangles.set_mask(
                TriAnalyzer(triangles).get_flat_tri_mask(mratio))
        triangles.set_mask(np.isnan(
            self.data.values if self.data.ndim == 1 else self.data[0].values))
        # in order to avoid lines spanning from right to left over the whole
        # plot due to the warping in cylindric (or quasi rectangular)
        # projections, we mask those triangles on the left and right side of
        # the globe
        wrap_proj_types = (ccrs._RectangularProjection,
                           ccrs._WarpedRectangularProjection,
                           ccrs.InterruptedGoodeHomolosine,
                           ccrs.Mercator)
        if isinstance(self.ax.projection, wrap_proj_types):
            lon_0 = self.ax.projection.proj4_params['lon_0']
            xmin = lon_0 - 100.
            xmax = lon_0 + 100.
            x = ccrs.PlateCarree(lon_0).transform_points(
                self.ax.projection, triangles.x, triangles.y)[:, 0]
            x = x[triangles.triangles]
            mask = np.array([
                np.any(arr > xmax) and np.any(arr < xmin) for arr in x])
            triangles.set_mask(mask)
        return triangles

    triangles = property(triangles, doc=DataGrid.triangles.__doc__)


class MapDensity(Density):
    """
    Change the density of the arrows

    Possible types
    --------------
    %(Density.possible_types)s"""

    def _set_quiver_density(self, value):
        if all(val == 1.0 for val in value):
            self.plot._kwargs.pop('regrid_shape', None)
        elif self.raw_data.decoder.is_unstructured(self.raw_data):
            warn("Quiver plot of unstructered data does not support the "
                 "density keyword!", PsyPlotRuntimeWarning, logger=self.logger)
        else:
            shape = self.data.shape[-2:]
            value = map(int, [value[0]*shape[0], value[1]*shape[1]])
            self.plot._kwargs['regrid_shape'] = tuple(value)

    def _unset_quiver_density(self):
        self.plot._kwargs.pop('regrid_shape', None)


class MapVectorColor(VectorColor):

    __doc__ = VectorColor.__doc__

    def _maybe_ravel(self, arr):
        # no need to ravel the data for quiver plots
        return np.asarray(arr)


class MapVectorPlot(VectorPlot):

    __doc__ = VectorPlot.__doc__

    dependencies = VectorPlot.dependencies + ['lonlatbox', 'transform', 'clon',
                                              'clat']

    def _get_data(self):
        data = self.data
        base_data = self.raw_data
        base_var = next(six.itervalues(base_data.base_variables))
        u, v = data[-2:].values
        x = base_data.decoder.get_x(base_var, coords=data.coords)
        y = base_data.decoder.get_y(base_var, coords=data.coords)
        x = x.values * 180. / np.pi if x.attrs.get('units') == 'radian' else \
            x.values
        y = y.values * 180. / np.pi if y.attrs.get('units') == 'radian' else \
            y.values
#        u = u.ravel()
#        v = v.ravel()
#        y = y.ravel()
#        x = x.ravel()
#        print u.shape, x.shape
#        x = x[~np.isnan(v)]
#        y = y[~np.isnan(v)]
#        u = u[~np.isnan(u)]
#        v = v[~np.isnan(v)]
        # we transform here manually because the transform keyword does not
        # work with unstructered grids and it sometimes shifts the plot data
        # for quiver and stream plots
        transform = self.transform.projection
        if x.ndim == 1 and u.ndim == 2:
            x, y = np.meshgrid(x, y)
        u, v = self.ax.projection.transform_vectors(transform, x, y, u, v)
        ret = self.ax.projection.transform_points(transform, x, y)
        x = ret[..., 0]
        y = ret[..., 1]
        self._kwargs.pop('transform', None)  # = self.ax.projection
        return x, y, u, v

    def _stream_plot(self):
        # update map extent such that it fits to the data limits (necessary
        # because streamplot scales the density based upon it). This however
        # does not work for matplotlib 1.5.0
        if not (mpl.__version__ == '1.5.0' and self.color.colored):
            self.ax.set_extent(self.lonlatbox.lonlatbox,
                               crs=ccrs.PlateCarree())
        else:
            self.ax.set_global()
        # Note that this method uses a bug fix through the
        # :class:`psyplot.plotter.colors.FixedColorMap` class
        x, y, u, v = self._get_data()
        self._plot = self.ax.streamplot(x, y, u, v, **self._kwargs)


class CombinedMapVectorPlot(MapVectorPlot):

    __doc__ = MapVectorPlot.__doc__

    def update(self, *args, **kwargs):
        self._kwargs['zorder'] = 2
        super(CombinedMapVectorPlot, self).update(*args, **kwargs)


class MapPlotter(Base2D):
    """Base plotter for visualizing data on a map
    """

    @property
    def ax(self):
        """Axes instance of the plot"""
        if self._ax is None:
            import matplotlib.pyplot as plt
            plt.figure()
            self._ax = plt.axes(projection=self.projection.projection)
        return self._ax

    @ax.setter
    def ax(self, value):
        self._ax = value

    _rcparams_string = ['plotter.maps.']

    transpose = None
    projection = Projection('projection')
    transform = Transform('transform')
    clon = CenterLon('clon')
    clat = CenterLat('clat')
    lonlatbox = LonLatBox('lonlatbox')
    lsm = LSM('lsm')
    grid_color = GridColor('grid_color')
    grid_labels = GridLabels('grid_labels')
    grid_labelsize = GridLabelSize('grid_labelsize')
    grid_settings = GridSettings('grid_settings')
    xgrid = XGrid('xgrid')
    ygrid = YGrid('ygrid')
    map_extent = MapExtent('map_extent')
    axiscolor = None  # the axiscolor formatoption does not really work
    datagrid = MapDataGrid('datagrid', index_in_list=0)

    @classmethod
    def _get_sample_projection(cls):
        """Returns None. May be subclassed to retumeshrn a projection that
        can be used when creating a subplot"""
        return ccrs.PlateCarree()


class FieldPlotter(Simple2DBase, MapPlotter, BasePlotter):
    """Plotter for 2D scalar fields on a map
    """
    plot = MapPlot2D('plot')


class VectorPlotter(MapPlotter, BaseVectorPlotter, BasePlotter):
    """Plotter for visualizing 2-dimensional vector data on a map

    See Also
    --------
    psyplot.plotter.simpleplotter.SimpleVectorPlotter:
        for a simple version of drawing vector data
    FieldPlotter: for plotting scaler fields
    CombinedPlotter: for combined scalar and vector fields
    """
    plot = MapVectorPlot('plot')
    density = MapDensity('density')
    color = MapVectorColor('color')


class CombinedPlotter(CombinedBase, FieldPlotter, VectorPlotter):
    """Combined 2D plotter and vector plotter on a map

    See Also
    --------
    psyplot.plotter.simpleplotter.CombinedSimplePlotter:
        for a simple version of this class
    FieldPlotter, VectorPlotter"""
    plot = MapPlot2D('plot', index_in_list=0)
    vplot = CombinedMapVectorPlot('vplot', cmap='vcmap', bounds='vbounds',
                                  index_in_list=1)
    density = MapDensity('density', plot='vplot', index_in_list=1)
    xgrid = XGrid('xgrid', index_in_list=1)
    ygrid = YGrid('ygrid', index_in_list=1)
