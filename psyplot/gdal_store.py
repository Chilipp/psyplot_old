# -*- coding: utf-8 -*-
"""Gdal Store for reading GeoTIFF files into an :class:`xray.Dataset`

This module contains the definition of the :class:`GdalStore` class that can
be used to read in a GeoTIFF file into an :class:`xray.Dataset`.
It requires that you have the python gdal module installed.

Examples
--------
to open a GeoTIFF file named ``'my_tiff.tiff'`` you can do::

    >>> from psyplot.gdal_store import GdalStore
    >>> from psyplot import open_dataset
    >>> ds = open_dataset(GdalStore('my_tiff'))

Or you use the `engine` of the :func:`psyplot.open_dataset` function:

    >>> ds = open_dataset('my_tiff.tiff', engine='gdal')"""
from gdal import Open, GetDataTypeName
from numpy import arange, nan, dtype
from xray import Variable
from collections import OrderedDict
from xray.core.utils import FrozenOrderedDict
from dask.array import Array
from xray.backends.common import AbstractDataStore
from .compat.pycompat import range


class GdalStore(AbstractDataStore):
    def __init__(self, filename):
        self.ds = Open(filename)
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
            dsk = {('x', 0, 0): (load, iband)}
            dt = dtype(GetDataTypeName(ds.GetRasterBand(iband).DataType))
            arr = Array(dsk, 'x', chunks, shape=shape, dtype=dt)
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
        lat = Array({('lat', 0): (load_lat,)}, 'lat', (self.ds.RasterYSize,),
                    shape=(self.ds.RasterYSize,), dtype=float)
        lon = Array({('lon', 0): (load_lon,)}, 'lon', (self.ds.RasterXSize,),
                    shape=(self.ds.RasterXSize,), dtype=float)
        return Variable(('lat',), lat), Variable(('lon',), lon)

    def get_attrs(self):
        return FrozenOrderedDict(self.ds.GetMetadata())
