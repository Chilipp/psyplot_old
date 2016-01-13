import os
from unittest import TestCase, main
from psyplot.data import CFDecoder, open_dataset
from psyplot.compat.pycompat import range
import _base_testing as bt
import numpy as np


class DecoderTest(TestCase):

    def test_1D_cf_bounds(self):
        final_bounds = np.arange(-180, 181, 30)
        lon = np.arange(-165, 166, 30)
        cf_bounds = np.zeros((len(lon), 2))
        for i in range(len(lon)):
            cf_bounds[i, :] = final_bounds[i:i+2]
        decoder = CFDecoder()
        self.assertEqual(list(final_bounds),
                         list(decoder._get_plotbounds_from_cf(lon, cf_bounds)))

    def test_1D_bounds_calculation(self):
        final_bounds = np.arange(-180, 181, 30)
        lon = np.arange(-165, 166, 30)
        decoder = CFDecoder()
        self.assertEqual(list(final_bounds),
                         list(decoder._infer_interval_breaks(lon)))

    def _test_dimname(self, func_name, name, uname=None):
        uname = uname or name
        ds = open_dataset(os.path.join(bt.test_dir, 'test-t2m-u-v.nc'))
        d = CFDecoder(ds)
        self.assertEqual(getattr(d, func_name)(ds.t2m), name)
        ds = open_dataset(os.path.join(bt.test_dir, 'icon_test.nc'))
        d = CFDecoder(ds)
        self.assertEqual(getattr(d, func_name)(ds.t2m), uname)

    def test_tname(self):
        """Test CFDecoder.get_tname method"""
        self._test_dimname('get_tname', 'time')

    def test_zname(self):
        """Test CFDecoder.get_zname method"""
        self._test_dimname('get_zname', 'lev')

    def test_xname(self):
        """Test CFDecoder.get_xname method"""
        self._test_dimname('get_xname', 'lon', 'ncells')

    def test_yname(self):
        """Test CFDecoder.get_yname method"""
        self._test_dimname('get_yname', 'lat', 'ncells')


if __name__ == '__main__':
    main()
