import os
import re
import unittest
import six
from functools import wraps
from itertools import starmap, repeat, chain
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from psyplot.plotter import _TempBool
from psyplot.plotter.maps import (
    FieldPlotter, VectorPlotter, rcParams, CombinedPlotter, InteractiveList)
import test_baseplotter as tb
import _base_testing as bt
import test_simpleplotter as ts
from psyplot import ArrayList, open_dataset
import psyplot.project as syp
from psyplot.compat.mplcompat import bold


class MapReferences(object):
    """Abstract base class for map reference plots"""

    def ref_lonlatbox(self, close=True):
        """Create reference file for lonlatbox formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.FieldPlotter.lonlatbox` formatoption"""
        sp = self.plot()
        sp.update(lonlatbox='Europe|India')
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('lonlatbox')))
        if close:
            sp.close(True, True)

    def ref_map_extent(self, close=True):
        """Create reference file for map_extent formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.FieldPlotter.map_extent` formatoption"""
        sp = self.plot()
        sp.update(map_extent='Europe|India')
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('map_extent')))
        if close:
            sp.close(True, True)

    def ref_lsm(self, close=True):
        """Create reference file for lsm formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.FieldPlotter.lsm` formatoption"""
        sp = self.plot()
        sp.update(lsm=False)
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('lsm')))
        if close:
            sp.close(True, True)

    def ref_projection(self, close=True):
        """Create reference file for projection formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.FieldPlotter.projection` formatoption"""
        import cartopy.crs as ccrs
        sp = self.plot()
        sp.update(projection='ortho', grid_labels=False)
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('projection1')))
        sp.update(projection='northpole')
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('projection2')))
        sp.update(projection=ccrs.LambertConformal())
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('projection3')))
        if close:
            sp.close(True, True)

    def ref_map_grid(self, close=True):
        """Create reference file for xgrid formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.FieldPlotter.xgrid` (and others)
        formatoption"""
        sp = self.plot()
        sp.update(xgrid=False, ygrid=False)
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('grid1')))
        sp.update(xgrid='rounded', ygrid=['data', 20])
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('grid2')))
        sp.update(xgrid=True, ygrid=True, grid_color='w')
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('grid3')))
        sp.update(xgrid=True, ygrid=True, grid_color='k', grid_settings={
            'linestyle': 'dashed'})
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('grid4')))
        if close:
            sp.close(True, True)


class FieldPlotterTest(tb.BasePlotterTest, ts.References2D, MapReferences):
    """Test :class:`psyplot.plotter.maps.FieldPlotter` class"""

    plot_type = 'map'

    def plot(self, **kwargs):
        name = kwargs.pop('name', 't2m')
        return syp.plot.mapplot(self.ncfile, name=name, **kwargs)

    @classmethod
    def setUpClass(cls):
        from matplotlib.backends.backend_pdf import PdfPages
        cls.ds = open_dataset(cls.ncfile)
        cls.data = ArrayList.from_dataset(
            cls.ds, t=0, z=0, name='t2m', auto_update=True)[0]
        cls.plotter = FieldPlotter(cls.data)
        if not os.path.isdir(bt.odir):
            os.makedirs(bt.odir)

        cls.pdf = PdfPages('test_%s.pdf' % cls.__name__)

    @unittest.skip("axiscolor formatoption not implemented")
    def test_axiscolor(self):
        pass

    def test_extend(self):
        """Test extend formatoption"""
        self.update(extend='both')
        self.assertEqual(self.plotter.cbar.cbars['b'].extend, 'both')
        self.update(extend='min')
        self.assertEqual(self.plotter.cbar.cbars['b'].extend, 'min')
        self.update(extend='neither')
        self.assertEqual(self.plotter.cbar.cbars['b'].extend, 'neither')

    @property
    def _minmax_cticks(self):
        return np.round(
            np.linspace(self.data.values.min(), self.data.values.max(), 11,
                        endpoint=True), decimals=2).tolist()

    def test_cticks(self):
        """Test cticks, cticksize, ctickweight, ctickprops formatoptions"""
        cticks = self._minmax_cticks
        self.update(cticks='minmax')
        cbar = self.plotter.cbar.cbars['b']
        self.assertEqual(list(map(
            lambda t: float(t.get_text()), cbar.ax.get_xticklabels())), cticks)
        self.update(cticklabels='%3.1f')
        cticks = np.round(cticks, decimals=1).tolist()
        self.assertEqual(list(map(
            lambda t: float(t.get_text()), cbar.ax.get_xticklabels())), cticks)
        self.update(cticksize=20, ctickweight='bold', ctickprops={
            'labelcolor': 'r'})
        texts = cbar.ax.get_xticklabels()
        n = len(texts)
        self.assertEqual([t.get_weight() for t in texts], [bold] * n)
        self.assertEqual([t.get_size() for t in texts], [20] * n)
        self.assertEqual([t.get_color() for t in texts], ['r'] * n)

    def test_clabel(self):
        """Test clabel, clabelsize, clabelweight, clabelprops formatoptions"""
        def get_clabel():
            return self.plotter.cbar.cbars['b'].ax.xaxis.get_label()
        self._label_test('clabel', get_clabel)
        label = get_clabel()
        self.update(clabelsize=22, clabelweight='bold',
                    clabelprops={'ha': 'left'})
        self.assertEqual(label.get_size(), 22)
        self.assertEqual(label.get_weight(), bold)
        self.assertEqual(label.get_ha(), 'left')

    def test_datagrid(self, *args):
        """Test datagrid formatoption"""
        self.update(lonlatbox='Europe', datagrid='k-')
        self.compare_figures(next(iter(args), self.get_ref_file('datagrid')))

    def test_cmap(self, *args):
        """Test colormap (cmap) formatoption"""
        self.update(cmap='RdBu')
        fname = next(iter(args), self.get_ref_file('cmap'))
        self.compare_figures(fname)
        self.update(cmap=plt.get_cmap('RdBu'))
        self.compare_figures(fname)

    def test_cbar(self, *args):
        """Test colorbar (cbar) formatoption"""
        self.update(cbar=['fb', 'fr', 'fl', 'ft', 'b', 'r'])
        self.compare_figures(next(iter(args), self.get_ref_file('cbar')))

    def test_bounds(self):
        """Test bounds formatoption"""
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(235, 310, 11, endpoint=True).tolist())
        self.update(bounds='minmax')
        bounds = [239.91, 246.89, 253.88, 260.87, 267.86, 274.84, 281.83,
                  288.82, 295.81, 302.79, 309.78]
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(), bounds)
        self.update(bounds=['rounded', 5, 5, 95])
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(245, 300, 5, endpoint=True).tolist())

    def test_miss_color(self, *args):
        """Test miss_color formatoption"""
        # We have to change the projection because cartopy (0.13.0) does not
        # support the :meth:`matplotlib.colors.ColorMap.set_bad` method for
        # rectangular projections
        self.update(maskless=280, miss_color='0.9', projection='ortho',
                    grid_labels=False)
        self.compare_figures(next(iter(args), self.get_ref_file('miss_color')))

    def test_cbarspacing(self, *args):
        """Test cbarspacing formatoption"""
        self.update(
            cbarspacing='proportional', cticks='rounded',
            bounds=list(range(235, 250)) + np.linspace(
                250, 295, 7, endpoint=True).tolist() + list(range(296, 310)))
        self.compare_figures(next(iter(args),
                                  self.get_ref_file('cbarspacing')))

    def test_lonlatbox(self, *args):
        """Test lonlatbox formatoption"""
        self.update(lonlatbox='Europe|India')
        ax = self.plotter.ax
        list(starmap(self.assertAlmostEqual, zip(
            ax.get_extent(ccrs.PlateCarree()), (-32.0, 97.0, -8.0, 81.0),
            repeat(5), repeat("Failed to set the extent to Europe and India!"))
            ))
        # test whether latitudes and longitudes succeded
        msg = "Failed to fit into lonlatbox limits for %s of %s."
        if isinstance(self.plotter.plot_data, InteractiveList):
            all_data = self.plotter.plot_data
        else:
            all_data = [self.plotter.plot_data]
        for data in all_data:
            self.assertGreaterEqual(data.lon.values.min(), -32.0,
                                    msg=msg % ('longitude', 'minimum'))
            self.assertLessEqual(data.lon.values.max(), 97.0,
                                 msg=msg % ('longitude', 'maximum'))
            self.assertGreaterEqual(data.lat.values.min(), -8.0,
                                    msg=msg % ('latitude', 'minimum'))
            self.assertLessEqual(data.lat.values.max(), 81.0,
                                 msg=msg % ('latitude', 'maximum'))
        self.compare_figures(next(iter(args), self.get_ref_file('lonlatbox')))

    def test_map_extent(self, *args):
        """Test map_extent formatoption"""
        self.update(map_extent='Europe|India')
        ax = self.plotter.ax
        list(starmap(self.assertAlmostEqual, zip(
            ax.get_extent(ccrs.PlateCarree()), (-32.0, 97.0, -8.0, 81.0),
            repeat(5), repeat("Failed to set the extent to Europe and India!"))
            ))
        self.compare_figures(next(iter(args), self.get_ref_file('map_extent')))

    def test_lsm(self, *args):
        """Test land-sea-mask formatoption"""
        self.update(lsm=False)
        self.compare_figures(next(iter(args), self.get_ref_file('lsm')))

    def test_projection(self, *args):
        """Test projection formatoption"""
        self.update(projection='ortho', grid_labels=False)
        self.compare_figures(next(iter(args),
                                  self.get_ref_file('projection1')))
        self.update(projection='northpole')
        self.compare_figures(next(iter(args),
                                  self.get_ref_file('projection2')))
        self.update(projection=ccrs.LambertConformal())
        self.compare_figures(next(iter(args),
                                  self.get_ref_file('projection3')))

    def test_grid(self, *args):
        """Test xgrid, ygrid, grid_color, grid_labels, grid_settings fmts"""
        self.update(xgrid=False, ygrid=False)
        self.compare_figures(next(iter(args), self.get_ref_file('grid1')))
        self.update(xgrid='rounded', ygrid=['data', 20])
        self.compare_figures(next(iter(args), self.get_ref_file('grid2')))
        self.update(xgrid=True, ygrid=True, grid_color='w')
        self.compare_figures(next(iter(args), self.get_ref_file('grid3')))
        self.update(xgrid=True, ygrid=True, grid_color='k',
                    grid_settings={'linestyle': 'dashed'})
        self.compare_figures(next(iter(args), self.get_ref_file('grid4')))

    def test_clon(self):
        """Test clon formatoption"""
        self.update(clon=180.)
        self.assertEqual(self.plotter.ax.projection.proj4_params['lon_0'],
                         180.)
        self.update(clon='India')
        self.assertEqual(self.plotter.ax.projection.proj4_params['lon_0'],
                         82.5)

    def test_clat(self):
        """Test clat formatoption"""
        self.update(projection='ortho', clat=60., grid_labels=False)
        self.assertEqual(self.plotter.ax.projection.proj4_params['lat_0'],
                         60.)
        self.update(clat='India')
        self.assertEqual(self.plotter.ax.projection.proj4_params['lat_0'],
                         13.5)

    def test_grid_labelsize(self):
        """Test grid_labelsize formatoption"""
        self.update(grid_labelsize=20)
        texts = list(chain(self.plotter.xgrid._gridliner.xlabel_artists,
                           self.plotter.ygrid._gridliner.ylabel_artists))
        self.assertEqual([t.get_size() for t in texts], [20] * len(texts))


class VectorPlotterTest(FieldPlotterTest, ts.References2D, MapReferences):
    """Test :class:`psyplot.plotter.maps.VectorPlotter` class"""

    plot_type = 'mapvector'

    def plot(self, **kwargs):
        color_fmts = syp.plot.mapvector.plotter_cls().fmt_groups['colors']
        fix_colorbar = not color_fmts.intersection(kwargs)
        kwargs.setdefault('color', 'absolute')
        kwargs.setdefault('lonlatbox', 'Europe')
        sp = syp.plot.mapvector(self.ncfile, name=[['u', 'v']], **kwargs)
        if fix_colorbar:
            # if we have no color formatoptions, we have to consider that
            # the position of the plot may have slighty changed
            sp.update(todefault=True, replot=True, **dict(
                item for item in kwargs.items() if item[0] != 'color'))
        return sp

    @unittest.skip("miss_color formatoption not implemented")
    def ref_miss_color(self, close=True):
        pass

    def ref_arrowsize(self, close=True):
        """Create reference file for arrowsize formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.VectorPlotter.arrowsize` (and others)
        formatoption"""
        sp = self.plot(arrowsize=100.0)
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('arrowsize')))
        if close:
            sp.close(True, True)

    def ref_density(self, close=True):
        """Create reference file for density formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.VectorPlotter.density` (and others)
        formatoption"""
        sp = self.plot(density=0.5)
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('density')))
        if close:
            sp.close(True, True)

    @classmethod
    def setUpClass(cls):
        from matplotlib.backends.backend_pdf import PdfPages
        cls.ds = open_dataset(cls.ncfile)
        rcParams[VectorPlotter().lonlatbox.default_key] = 'Europe'
        cls.data = ArrayList.from_dataset(
            cls.ds, t=0, z=0, name=[['u', 'v']], auto_update=True)[0]
        cls.data.attrs['long_name'] = 'absolute wind speed'
        cls.data.name = 'wind'
        cls.plotter = VectorPlotter(cls.data)
        if not os.path.isdir(bt.odir):
            os.makedirs(bt.odir)

        cls.pdf = PdfPages('test_%s.pdf' % cls.__name__)
        cls._color_fmts = cls.plotter.fmt_groups['colors']

        # there is an issue with the colorbar that the size of the axes changes
        # slightly after replotting. Therefore we force a replot here
        cls.plotter.update(color='absolute')
        cls.plotter.update(todefault=True, replot=True)

    def update(self, *args, **kwargs):
        if self._color_fmts.intersection(kwargs) or any(
                re.match('ctick|clabel', fmt) for fmt in kwargs):
            kwargs.setdefault('color', 'absolute')
        super(VectorPlotterTest, self).update(*args, **kwargs)

    @unittest.skip("Not supported")
    def test_maskless(self):
        pass

    @unittest.skip("Not supported")
    def test_maskgreater(self):
        pass

    @unittest.skip("Not supported")
    def test_maskleq(self):
        pass

    @unittest.skip("Not supported")
    def test_maskgeq(self):
        pass

    @unittest.skip("Not supported")
    def test_maskbetween(self):
        pass

    @unittest.skip("Not supported")
    def test_miss_color(self):
        pass

    def test_bounds(self):
        """Test bounds formatoption"""
        self.update(color='absolute')
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(0, 15, 11, endpoint=True).tolist())
        self.update(bounds='minmax')
        bounds = [0.36, 1.52, 2.68, 3.85, 5.01, 6.17, 7.33, 8.5, 9.66, 10.82,
                  11.99]
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(), bounds)
        self.update(bounds=['rounded', 5, 5, 95])
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(1.0, 10.0, 5, endpoint=True).tolist())

    def test_cbarspacing(self, *args):
        """Test cbarspacing formatoption"""
        self.update(
            cbarspacing='proportional', cticks='rounded', color='absolute',
            bounds=np.arange(0, 1.45, 0.1).tolist() + np.linspace(
                    1.5, 13.5, 7, endpoint=True).tolist() + np.arange(
                        13.6, 15.05, 0.1).tolist())
        self.compare_figures(next(iter(args),
                                  self.get_ref_file('cbarspacing')))

    def test_arrowsize(self, *args):
        """Test arrowsize formatoption"""
        self.update(arrowsize=100.0)
        self.compare_figures(next(iter(args), self.get_ref_file('arrowsize')))

    @unittest.skipIf(
        six.PY34, "The axes size changes when using the density formatoption")
    def test_density(self, *args):
        """Test density formatoption"""
        self.update(density=0.5)
        self.compare_figures(next(iter(args), self.get_ref_file('density')))

    @property
    def _minmax_cticks(self):
        speed = (self.plotter.plot_data.values[0]**2 +
                 self.plotter.plot_data.values[1]**2) ** 0.5
        speed = speed[~np.isnan(speed)]
        return np.round(
            np.linspace(speed.min(), speed.max(), 11, endpoint=True),
            decimals=2).tolist()


class StreamVectorPlotterTest(VectorPlotterTest):
    """Test case for stream plot of :class:`psyplot.plotter.maps.VectorPlotter`
    """

    @classmethod
    def setUpClass(cls):
        rcParams[VectorPlotter().plot.default_key] = 'stream'
        return super(cls, cls).setUpClass()

    def get_ref_file(self, identifier):
        return super(StreamVectorPlotterTest, self).get_ref_file(
            identifier + '_stream')

    def ref_arrowsize(self, *args):
        """Create reference file for arrowsize formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.VectorPlotter.arrowsize` (and others)
        formatoption"""
        sp = self.plot(arrowsize=2.0)
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('arrowsize')))

    def ref_arrowstyle(self, *args):
        """Create reference file for arrowstyle formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.VectorPlotter.arrowstyle` (and others)
        formatoption"""
        sp = self.plot(arrowsize=2.0, arrowstyle='fancy')
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('arrowstyle')))

    def test_arrowsize(self, *args):
        """Test arrowsize formatoption"""
        self.update(arrowsize=2.0)
        self.compare_figures(next(iter(args), self.get_ref_file('arrowsize')))

    def test_arrowstyle(self, *args):
        """Test arrowstyle formatoption"""
        self.update(arrowsize=2.0, arrowstyle='fancy')
        self.compare_figures(next(iter(args), self.get_ref_file('arrowstyle')))

    def test_density(self, *args):
        """Test density formatoption"""
        self.update(density=0.5)
        self.compare_figures(next(iter(args), self.get_ref_file('density')))


def _do_from_both(func):
    """Call the given `func` only from :class:`FieldPlotterTest and
    :class:`VectorPlotterTest`"""
    func.__doc__ = getattr(VectorPlotterTest, func.__name__).__doc__

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        getattr(FieldPlotterTest, func.__name__)(self, *args, **kwargs)
        if hasattr(self, 'plotter'):
            self.plotter.update(todefault=True)
        with self.vector_mode:
            getattr(VectorPlotterTest, func.__name__)(self, *args, **kwargs)

    return wrapper


def _in_vector_mode(func):
    """Call the given `func` only from :class:`FieldPlotterTest and
    :class:`VectorPlotterTest`"""
    func.__doc__ = getattr(VectorPlotterTest, func.__name__).__doc__

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.vector_mode:
            getattr(VectorPlotterTest, func.__name__)(self, *args, **kwargs)

    return wrapper


class _CombinedPlotterData(object):
    """Descriptor that returns the data"""
    # Note: We choose to use a descriptor rather than a usual property because
    # it shall also work for class objects and not only instances

    def __get__(self, instance, owner):
        if instance is None:
            return owner._data
        if instance.vector_mode:
            return instance._data[1]
        return instance._data[0]

    def __set__(self, instance, value):
        instance._data = value


class CombinedPlotterTest(VectorPlotterTest):
    """Test case for stream plot of :class:`psyplot.plotter.maps.CombinedPlotter`
    """

    plot_type = 'mapcombined'

    data = _CombinedPlotterData()

    @property
    def vector_mode(self):
        """:class:`bool` indicating whether a vector specific formatoption is
        tested or not"""
        try:
            return self._vector_mode
        except AttributeError:
            self._vector_mode = _TempBool(False)
            return self._vector_mode

    @vector_mode.setter
    def vector_mode(self, value):
        self.vector_mode.value = bool(value)

    def compare_figures(self, fname, **kwargs):
        kwargs.setdefault('tol', 10)
        return super(CombinedPlotterTest, self).compare_figures(
            fname, **kwargs)

    @classmethod
    def setUpClass(cls):
        from matplotlib.backends.backend_pdf import PdfPages
        cls.ds = open_dataset(cls.ncfile)
        rcParams[CombinedPlotter().lonlatbox.default_key] = 'Europe'
        rcParams[CombinedPlotter().vcmap.default_key] = 'winter'
        cls._data = ArrayList.from_dataset(
            cls.ds, t=0, z=0, name=[['t2m', ['u', 'v']]], auto_update=True,
            prefer_list=True)[0]
        cls._data.attrs['long_name'] = 'Temperature'
        cls._data.attrs['name'] = 't2m'
        cls.plotter = CombinedPlotter(cls.data)
        if not os.path.isdir(bt.odir):
            os.makedirs(bt.odir)

        cls.pdf = PdfPages('test_%s.pdf' % cls.__name__)
        cls._color_fmts = cls.plotter.fmt_groups['colors']

        # there is an issue with the colorbar that the size of the axes changes
        # slightly after replotting. Therefore we force a replot here
        cls.plotter.update(color='absolute')
        cls.plotter.update(todefault=True, replot=True)

    def plot(self, **kwargs):
        color_fmts = syp.plot.mapvector.plotter_cls().fmt_groups['colors']
        fix_colorbar = not color_fmts.intersection(kwargs)
        kwargs.setdefault('lonlatbox', 'Europe')
        kwargs.setdefault('color', 'absolute')
        if self.vector_mode:
            kwargs = self._rename_fmts(kwargs)
        sp = syp.plot.mapcombined(self.ncfile, name=[['t2m', ['u', 'v']]],
                                  **kwargs)
        if not self.vector_mode or fix_colorbar:
            # if we have no color formatoptions, we have to consider that
            # the position of the plot may have slighty changed
            sp.update(todefault=True, replot=True, **dict(
                item for item in kwargs.items() if item[0] != 'color'))
        return sp

    def _rename_fmts(self, kwargs):
        def check_key(key):
            if not any(re.match('v' + key, fmt) for fmt in vcolor_fmts):
                return key
            else:
                return 'v' + key
        vcolor_fmts = {
            fmt for fmt in chain(
                syp.plot.mapcombined.plotter_cls().fmt_groups['colors'],
                ['ctick|clabel']) if fmt.startswith('v')}
        return {check_key(key): val for key, val in kwargs.items()}

    def update(self, *args, **kwargs):
        if self.vector_mode and (
                self._color_fmts.intersection(kwargs) or any(
                    re.match('ctick|clabel', fmt) for fmt in kwargs)):
            kwargs.setdefault('color', 'absolute')
            kwargs = self._rename_fmts(kwargs)
        super(VectorPlotterTest, self).update(*args, **kwargs)

    def get_ref_file(self, identifier):
        if self.vector_mode:
            identifier += '_vector'
        return super(CombinedPlotterTest, self).get_ref_file(identifier)

    @property
    def _minmax_cticks(self):
        if not self.vector_mode:
            return np.round(
                np.linspace(self.plotter.plot_data[0].values.min(),
                            self.plotter.plot_data[0].values.max(), 11,
                            endpoint=True), decimals=2).tolist()
        speed = (self.plotter.plot_data[1].values[0]**2 +
                 self.plotter.plot_data[1].values[1]**2) ** 0.5
        return np.round(
            np.linspace(speed.min(), speed.max(), 11, endpoint=True),
            decimals=2).tolist()

    def ref_density(self, close=True):
        """Create reference file for density formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.maps.VectorPlotter.density` (and others)
        formatoption"""
        # we have to make sure, that the color keyword is not set to 'absolute'
        # because it does not work for quiver plots
        sp = self.plot(density=0.5, color='k')
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('density')))
        if close:
            sp.close(True, True)

    @_do_from_both
    def ref_cbar(self, close=True):
        pass

    @_do_from_both
    def ref_cbarspacing(self, close=True):
        pass

    @_do_from_both
    def ref_cmap(self, close=True):
        pass

    def ref_miss_color(self, close=True):
        FieldPlotterTest.ref_miss_color(self, close)

    @_in_vector_mode
    def ref_arrowsize(self, *args, **kwargs):
        pass

    def _label_test(self, key, label_func):
        kwargs = {
            key: "Test plot at %Y-%m-%d, {tinfo} o'clock of %(long_name)s"}
        self.update(**kwargs)
        self.assertEqual(
            u"Test plot at 1979-01-31, 18:00 o'clock of %s" % (
                self.data.attrs.get('long_name', 'Temperature')),
            label_func().get_text())
        self._data.update(time=1)
        self.assertEqual(
            u"Test plot at 1979-02-28, 18:00 o'clock of %s" % (
                self.data.attrs.get('long_name', 'Temperature')),
            label_func().get_text())
        self._data.update(time=0)

    def test_miss_color(self, *args, **kwargs):
        FieldPlotterTest.test_miss_color(self, *args, **kwargs)

    @_do_from_both
    def test_cbar(self, *args, **kwargs):
        pass

    @_do_from_both
    def test_cbarspacing(self, *args, **kwargs):
        pass

    @_do_from_both
    def test_cmap(self, *args, **kwargs):
        pass

    @_in_vector_mode
    def test_arrowsize(self):
        pass

    def test_bounds(self):
        """Test bounds formatoption"""
        # test bounds of scalar field
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(245, 290, 11, endpoint=True).tolist())
        self.update(bounds='minmax')
        bounds = [248.07, 252.24, 256.42, 260.6, 264.78, 268.95, 273.13,
                  277.31, 281.48, 285.66, 289.84]
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(), bounds)
        self.update(bounds=['rounded', 5, 5, 95])
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(250, 290, 5, endpoint=True).tolist())

        # test vector bounds
        self.update(color='absolute')
        self.assertEqual(
            np.round(self.plotter.vbounds.norm.boundaries, 2).tolist(),
            np.linspace(0, 15, 11, endpoint=True).tolist())
        self.update(vbounds='minmax')
        bounds = [0.36, 1.52, 2.68, 3.85, 5.01, 6.17, 7.33, 8.5, 9.66, 10.82,
                  11.99]
        self.assertEqual(
            np.round(self.plotter.vbounds.norm.boundaries, 2).tolist(), bounds)
        self.update(vbounds=['rounded', 5, 5, 95])
        self.assertEqual(
            np.round(self.plotter.vbounds.norm.boundaries, 2).tolist(),
            np.linspace(1.0, 10.0, 5, endpoint=True).tolist())

    def test_clabel(self):
        def get_clabel():
            return self.plotter.vcbar.cbars['b'].ax.xaxis.get_label()
        FieldPlotterTest.test_clabel(self)
        with self.vector_mode:
            self.update(color='absolute')
            self._label_test('vclabel', get_clabel)
            label = get_clabel()
            self.update(vclabelsize=22, vclabelweight='bold',
                        vclabelprops={'ha': 'left'})
            self.assertEqual(label.get_size(), 22)
            self.assertEqual(label.get_weight(), bold)
            self.assertEqual(label.get_ha(), 'left')


class IconFieldPlotterTest(FieldPlotterTest):
    """Test :class:`psyplot.plotter.maps.FieldPlotter` class for icon grid"""

    grid_type = 'icon'

    ncfile = 'icon_test.nc'

    def test_bounds(self):
        """Test bounds formatoption"""
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(240, 310, 11, endpoint=True).tolist())
        self.update(bounds='minmax')
        bounds = [241.19, 248., 254.81, 261.62, 268.43, 275.24, 282.05, 288.86,
                  295.67, 302.48, 309.29]
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(), bounds)
        self.update(bounds=['rounded', 5, 5, 95])
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(250, 305, 5, endpoint=True).tolist())

    def test_lonlatbox(self, *args):
        """Test lonlatbox formatoption"""
        def get_unmasked(coord):
            """return the values of the coordinate that is not masked in the
            data"""
            return coord.values[~np.isnan(data.values)]
        self.update(lonlatbox='Europe|India')
        ax = self.plotter.ax
        list(starmap(self.assertAlmostEqual, zip(
            ax.get_extent(ccrs.PlateCarree()), (-32.0, 97.0, -8.0, 81.0),
            repeat(5), repeat("Failed to set the extent to Europe and India!"))
            ))
        # test whether latitudes and longitudes succeded
        msg = "Failed to fit into lonlatbox limits for %s of %s."
        if isinstance(self.plotter.plot_data, InteractiveList):
            all_data = self.plotter.plot_data
        else:
            all_data = [self.plotter.plot_data]
        for data in all_data:
            self.assertGreaterEqual(get_unmasked(data.clon).min(), -32.0,
                                    msg=msg % ('longitude', 'minimum'))
            self.assertLessEqual(get_unmasked(data.clon).max(), 97.0,
                                 msg=msg % ('longitude', 'maximum'))
            self.assertGreaterEqual(get_unmasked(data.clat).min(), -8.0,
                                    msg=msg % ('latitude', 'minimum'))
            self.assertLessEqual(get_unmasked(data.clat).max(), 81.0,
                                 msg=msg % ('latitude', 'maximum'))
        self.compare_figures(next(iter(args), self.get_ref_file('lonlatbox')))

    @unittest.skip(
        'miss_color formatoption does not work for unstructered data')
    def test_miss_color(self, *args):
        """Test miss_color formatoption"""
        pass

    @unittest.skip(
        'miss_color formatoption does not work for unstructered data')
    def ref_miss_color(self, *args):
        """Test miss_color formatoption"""
        pass


class IconVectorPlotterTest(VectorPlotterTest):
    """Test :class:`psyplot.plotter.maps.VectorPlotter` class for icon grid"""

    grid_type = 'icon'

    ncfile = 'icon_test.nc'

    def test_lonlatbox(self, *args):
        """Test lonlatbox formatoption"""
        def get_unmasked(coord):
            """return the values of the coordinate that is not masked in the
            data"""
            return coord.values[~np.isnan(data[0].values)]
        self.update(lonlatbox='Europe|India')
        ax = self.plotter.ax
        list(starmap(self.assertAlmostEqual, zip(
            ax.get_extent(ccrs.PlateCarree()), (-32.0, 97.0, -8.0, 81.0),
            repeat(5), repeat("Failed to set the extent to Europe and India!"))
            ))
        # test whether latitudes and longitudes succeded
        msg = "Failed to fit into lonlatbox limits for %s of %s."
        if isinstance(self.plotter.plot_data, InteractiveList):
            all_data = self.plotter.plot_data
        else:
            all_data = [self.plotter.plot_data]
        for data in all_data:
            self.assertGreaterEqual(get_unmasked(data.clon).min(), -32.0,
                                    msg=msg % ('longitude', 'minimum'))
            self.assertLessEqual(get_unmasked(data.clon).max(), 97.0,
                                 msg=msg % ('longitude', 'maximum'))
            self.assertGreaterEqual(get_unmasked(data.clat).min(), -8.0,
                                    msg=msg % ('latitude', 'minimum'))
            self.assertLessEqual(get_unmasked(data.clat).max(), 81.0,
                                 msg=msg % ('latitude', 'maximum'))
        self.compare_figures(next(iter(args), self.get_ref_file('lonlatbox')))

    @unittest.skip("Density for quiver plots of unstructered data is not "
                   "supported!")
    def ref_density(self):
        pass

    @unittest.skip("Density for quiver plots of unstructered data is not "
                   "supported!")
    def test_density(self):
        pass

    def test_bounds(self):
        """Test bounds formatoption"""
        self.update(color='absolute')
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(0, 15, 11, endpoint=True).tolist())
        self.update(bounds='minmax')
        bounds = [0.13, 1.2, 2.27, 3.34, 4.41, 5.48, 6.55, 7.62, 8.69, 9.76,
                  10.83]
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(), bounds)
        self.update(bounds=['rounded', 5, 5, 95])
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.round(np.linspace(1.0, 9.5, 5, endpoint=True), 2).tolist())


class IconCombinedPlotterTest(CombinedPlotterTest):
    """Test :class:`psyplot.plotter.maps.CombinedPlotter` class for icon grid
    """

    grid_type = 'icon'

    ncfile = 'icon_test.nc'

    @unittest.skip("Density for quiver plots of unstructered data is not "
                   "supported!")
    def ref_density(self):
        pass

    @unittest.skip("Density for quiver plots of unstructered data is not "
                   "supported!")
    def test_density(self):
        pass

    def test_bounds(self):
        """Test bounds formatoption"""
        # test bounds of scalar field
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(250, 290, 11, endpoint=True).tolist())
        self.update(bounds='minmax')
        bounds = [252.64, 256.24, 259.84, 263.44, 267.04, 270.64, 274.24,
                  277.84, 281.44, 285.04, 288.64]
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(), bounds)
        self.update(bounds=['rounded', 5, 5, 95])
        self.assertEqual(
            np.round(self.plotter.bounds.norm.boundaries, 2).tolist(),
            np.linspace(255, 290, 5, endpoint=True).tolist())

        # test vector bounds
        self.update(color='absolute')
        self.assertEqual(
            np.round(self.plotter.vbounds.norm.boundaries, 2).tolist(),
            np.linspace(0, 15, 11, endpoint=True).tolist())
        self.update(vbounds='minmax')
        bounds = [0.13, 1.2, 2.27, 3.34, 4.41, 5.48, 6.55, 7.62, 8.69, 9.76,
                  10.83]
        self.assertEqual(
            np.round(self.plotter.vbounds.norm.boundaries, 2).tolist(), bounds)
        self.update(vbounds=['rounded', 5, 5, 95])
        self.assertEqual(
            np.round(self.plotter.vbounds.norm.boundaries, 2).tolist(),
            np.round(np.linspace(1.0, 9.5, 5, endpoint=True), 2).tolist())

    def test_lonlatbox(self, *args):
        """Test lonlatbox formatoption"""
        def get_unmasked(coord):
            """return the values of the coordinate that is not masked in the
            data"""
            arr = data if data.ndim == 1 else data[0]
            return coord.values[~np.isnan(arr.values)]
        self.update(lonlatbox='Europe|India')
        ax = self.plotter.ax
        list(starmap(self.assertAlmostEqual, zip(
            ax.get_extent(ccrs.PlateCarree()), (-32.0, 97.0, -8.0, 81.0),
            repeat(5), repeat("Failed to set the extent to Europe and India!"))
            ))
        # test whether latitudes and longitudes succeded
        msg = "Failed to fit into lonlatbox limits for %s of %s."
        if isinstance(self.plotter.plot_data, InteractiveList):
            all_data = self.plotter.plot_data
        else:
            all_data = [self.plotter.plot_data]
        for data in all_data:
            self.assertGreaterEqual(get_unmasked(data.clon).min(), -32.0,
                                    msg=msg % ('longitude', 'minimum'))
            self.assertLessEqual(get_unmasked(data.clon).max(), 97.0,
                                 msg=msg % ('longitude', 'maximum'))
            self.assertGreaterEqual(get_unmasked(data.clat).min(), -8.0,
                                    msg=msg % ('latitude', 'minimum'))
            self.assertLessEqual(get_unmasked(data.clat).max(), 81.0,
                                 msg=msg % ('latitude', 'maximum'))
        self.compare_figures(next(iter(args), self.get_ref_file('lonlatbox')))

    @unittest.skip(
        'miss_color formatoption does not work for unstructered data')
    def test_miss_color(self, *args):
        """Test miss_color formatoption"""
        pass

    @unittest.skip(
        'miss_color formatoption does not work for unstructered data')
    def ref_miss_color(self, *args):
        """Test miss_color formatoption"""
        pass

    @property
    def _minmax_cticks(self):
        if not self.vector_mode:
            arr = self.plotter.plot_data[0].values
            arr = arr[~np.isnan(arr)]
            return np.round(
                np.linspace(arr.min(), arr.max(), 11, endpoint=True),
                decimals=2).tolist()
        arr = self.plotter.plot_data[1].values
        speed = (arr[0]**2 + arr[1]**2) ** 0.5
        speed = speed[~np.isnan(speed)]
        return np.round(
            np.linspace(speed.min(), speed.max(), 11, endpoint=True),
            decimals=2).tolist()


if __name__ == '__main__':
    bt.RefTestProgram()
