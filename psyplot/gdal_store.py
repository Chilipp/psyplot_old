# -*- coding: utf-8 -*-
"""Gdal Store for reading GeoTIFF files into an :class:`xarray.Dataset`

This module contains the definition of the :class:`GdalStore` class that can
be used to read in a GeoTIFF file into an :class:`xarray.Dataset`.
It requires that you have the python gdal module installed.

Examples
--------
to open a GeoTIFF file named ``'my_tiff.tiff'`` you can do::

    >>> from psyplot.gdal_store import GdalStore
    >>> from xarray import open_dataset
    >>> ds = open_dataset(GdalStore('my_tiff'))

Or you use the `engine` of the :func:`psyplot.open_dataset` function:

    >>> ds = open_dataset('my_tiff.tiff', engine='gdal')"""
from numpy import arange, nan, dtype
from xarray import Variable
from collections import OrderedDict
from xarray.core.utils import FrozenOrderedDict
from xarray.backends.common import AbstractDataStore
from psyplot.compat.pycompat import range
try:
    import gdal
    from osgeo import gdal_array
except ImportError as e:
    from .data import _MissingModule
    gdal = _MissingModule(e)
try:
    from dask.array import Array
    with_dask = True
except ImportError:
    with_dask = False


class GdalStore(AbstractDataStore):
    """Datastore to read raster files suitable for the gdal package

    We recommend to use the :func:`psyplot.open_dataset` function to open
    a geotiff file::

        >>> ds = psyplot.open_dataset('my_geotiff.tiff', engine='gdal')"""

    def __init__(self, filename):
        """
        Parameters
        ----------
        filename: str
            The path to the GeoTIFF file"""
        self.ds = gdal.Open(filename)
        self._filename = filename

    def get_variables(self):  # pragma: no cover
        def load(band):
            band = ds.GetRasterBand(band)
            a = band.ReadAsArray()
            no_data = band.GetNoDataValue()
            if no_data is not None:
                try:
                    a[a == no_data] = a.dtype.type(nan)
                except ValueError:
                    pass
            return a
        ds = self.ds
        dims = ['lat', 'lon']
        chunks = ((ds.RasterYSize,), (ds.RasterXSize,))
        shape = (ds.RasterYSize, ds.RasterXSize)
        variables = OrderedDict()
        for iband in range(1, ds.RasterCount+1):
            if with_dask:
                dsk = {('x', 0, 0): (load, iband)}
                dt = dtype(gdal_array.codes[ds.GetRasterBand(iband).DataType])
                arr = Array(dsk, 'x', chunks, shape=shape, dtype=dt)
            else:
                arr = load(iband)
            try:
                dt.type(nan)
                attrs = {'_FillValue': nan}
            except ValueError:
                no_data = ds.GetRasterBand(iband).GetNoDataValue()
                attrs = {'_FillValue': no_data} if no_data else {}
            variables['band%i' % iband] = Variable(dims, arr, attrs)
        variables['lat'], variables['lon'] = self._load_GeoTransform()
        return FrozenOrderedDict(variables)

    def _load_GeoTransform(self):
        """Calculate latitude and longitude variable calculated from the
        gdal.Open.GetGeoTransform method"""
        def load_lon():
            return arange(ds.RasterXSize)*b[1]+b[0]

        def load_lat():
            return arange(ds.RasterYSize)*b[5]+b[3]
        ds = self.ds
        b = self.ds.GetGeoTransform()  # bbox, interval
        if with_dask:
            lat = Array(
                {('lat', 0): (load_lat,)}, 'lat', (self.ds.RasterYSize,),
                shape=(self.ds.RasterYSize,), dtype=float)
            lon = Array(
                {('lon', 0): (load_lon,)}, 'lon', (self.ds.RasterXSize,),
                shape=(self.ds.RasterXSize,), dtype=float)
        else:
            lat = load_lat()
            lon = load_lon()
        return Variable(('lat',), lat), Variable(('lon',), lon)

    def get_attrs(self):
        return FrozenOrderedDict(self.ds.GetMetadata())
