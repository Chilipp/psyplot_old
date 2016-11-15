"""Test module of the :mod:`psyplot.data` module"""
import os
from unittest import TestCase, main
from psyplot.compat.pycompat import range
import psyplot.data as psyd
import _base_testing as bt
import numpy as np


class DecoderTest(TestCase):

    def test_1D_cf_bounds(self):
        final_bounds = np.arange(-180, 181, 30)
        lon = np.arange(-165, 166, 30)
        cf_bounds = np.zeros((len(lon), 2))
        for i in range(len(lon)):
            cf_bounds[i, :] = final_bounds[i:i+2]
        decoder = psyd.CFDecoder()
        self.assertEqual(list(final_bounds),
                         list(decoder._get_plotbounds_from_cf(lon, cf_bounds)))

    def test_1D_bounds_calculation(self):
        final_bounds = np.arange(-180, 181, 30)
        lon = np.arange(-165, 166, 30)
        decoder = psyd.CFDecoder()
        self.assertEqual(list(final_bounds),
                         list(decoder._infer_interval_breaks(lon)))

    def _test_dimname(self, func_name, name, uname=None, name2d=False,
                      circ_name=None):
        def check_ds(name):
            self.assertEqual(getattr(d, func_name)(ds.t2m), name)
            if name2d:
                self.assertEqual(getattr(d, func_name)(ds.t2m_2d), name)
            else:
                self.assertIsNone(getattr(d, func_name)(ds.t2m_2d))
        uname = uname or name
        circ_name = circ_name or name
        ds = psyd.open_dataset(os.path.join(bt.test_dir, 'test-t2m-u-v.nc'))
        d = psyd.CFDecoder(ds)
        check_ds(name)
        ds.close()
        ds = psyd.open_dataset(os.path.join(bt.test_dir, 'icon_test.nc'))
        d = psyd.CFDecoder(ds)
        check_ds(uname)
        ds.close()
        ds = psyd.open_dataset(
            os.path.join(bt.test_dir, 'circumpolar_test.nc'))
        d = psyd.CFDecoder(ds)
        check_ds(circ_name)
        ds.close()

    def _test_coord(self, func_name, name, uname=None, name2d=False,
                    circ_name=None):
        def check_ds(name):
            self.assertEqual(getattr(d, func_name)(ds.t2m).name, name)
            if name2d:
                self.assertEqual(getattr(d, func_name)(ds.t2m_2d).name, name)
            else:
                self.assertIsNone(getattr(d, func_name)(ds.t2m_2d))
        uname = uname or name
        circ_name = circ_name or name
        ds = psyd.open_dataset(os.path.join(bt.test_dir, 'test-t2m-u-v.nc'))
        d = psyd.CFDecoder(ds)
        check_ds(name)
        ds.close()
        ds = psyd.open_dataset(os.path.join(bt.test_dir, 'icon_test.nc'))
        d = psyd.CFDecoder(ds)
        check_ds(uname)
        ds.close()
        ds = psyd.open_dataset(
            os.path.join(bt.test_dir, 'circumpolar_test.nc'))
        d = psyd.CFDecoder(ds)
        check_ds(circ_name)
        ds.close()

    def test_tname(self):
        """Test CFDecoder.get_tname method"""
        self._test_dimname('get_tname', 'time')

    def test_zname(self):
        """Test CFDecoder.get_zname method"""
        self._test_dimname('get_zname', 'lev')

    def test_xname(self):
        """Test CFDecoder.get_xname method"""
        self._test_dimname('get_xname', 'lon', 'ncells', True,
                           circ_name='x')

    def test_yname(self):
        """Test CFDecoder.get_yname method"""
        self._test_dimname('get_yname', 'lat', 'ncells', True,
                           circ_name='y')

    def test_t(self):
        """Test CFDecoder.get_t method"""
        self._test_coord('get_t', 'time')

    def test_z(self):
        """Test CFDecoder.get_z method"""
        self._test_coord('get_z', 'lev')

    def test_x(self):
        """Test CFDecorder.get_x method"""
        self._test_coord('get_x', 'lon', 'clon', True,
                         circ_name='longitude')

    def test_y(self):
        """Test CFDecoder.get_y method"""
        self._test_coord('get_y', 'lat', 'clat', True,
                         circ_name='latitude')

    def test_idims(self):
        ds = psyd.open_dataset(os.path.join(bt.test_dir, 'test-t2m-u-v.nc'))
        arr = ds.t2m[1:, 1]
        arr.psy.init_accessor(base=ds)
        dims = arr.psy.idims
        for dim in ['time', 'lev', 'lat', 'lon']:
            self.assertEqual(
                psyd.safe_list(ds[dim][dims[dim]]),
                psyd.safe_list(arr.coords[dim]),
                msg="Slice %s for dimension %s is wrong!" % (dims[dim], dim))


if __name__ == '__main__':
    main()
