"""Test module of the :mod:`psyplot.data` module"""
import os
import six
import unittest
import pandas as pd
import xarray as xr
from psyplot.compat.pycompat import range
import psyplot.data as psyd
import _base_testing as bt
import numpy as np


class DecoderTest(unittest.TestCase):
    """Test the :class:`psyplot.data.CFDecoder` class"""

    def test_1D_cf_bounds(self):
        """Test whether the CF Conventions for 1D bounaries are correct"""
        final_bounds = np.arange(-180, 181, 30)
        lon = xr.Variable(('lon', ), np.arange(-165, 166, 30),
                          {'bounds': 'lon_bounds'})
        cf_bounds = xr.Variable(('lon', 'bnds'), np.zeros((len(lon), 2)))
        for i in range(len(lon)):
            cf_bounds[i, :] = final_bounds[i:i+2]
        ds = xr.Dataset(coords={'lon': lon, 'lon_bounds': cf_bounds})
        decoder = psyd.CFDecoder(ds)
        self.assertEqual(list(final_bounds),
                         list(decoder.get_plotbounds(lon)))

    def test_1D_bounds_calculation(self):
        """Test whether the 1D cell boundaries are calculated correctly"""
        final_bounds = np.arange(-180, 181, 30)
        lon = xr.Variable(('lon', ), np.arange(-165, 166, 30))
        ds = xr.Dataset(coords={'lon': lon})
        decoder = psyd.CFDecoder(ds)
        self.assertEqual(list(final_bounds),
                         list(decoder.get_plotbounds(lon)))

    def _test_dimname(self, func_name, name, uname=None, name2d=False,
                      circ_name=None):
        def check_ds(name):
            self.assertEqual(getattr(d, func_name)(ds.t2m), name)
            self.assertEqual(getattr(d, func_name)(ds.t2m,
                             coords=ds.t2m.coords), name)
            if name2d:
                self.assertEqual(getattr(d, func_name)(ds.t2m_2d), name)
            else:
                self.assertIsNone(getattr(d, func_name)(ds.t2m_2d))
            if six.PY3:
                # Test whether the warning is raised if the decoder finds
                # multiple dimensions
                with self.assertWarnsRegex(RuntimeWarning,
                                           'multiple matches'):
                    coords = 'time lat lon lev x y latitude longitude'.split()
                    ds.t2m.attrs.pop('coordinates', None)
                    for dim in 'xytz':
                        getattr(d, dim).update(coords)
                    for coord in set(coords).intersection(ds.coords):
                        ds.coords[coord].attrs.pop('axis', None)
                    getattr(d, func_name)(ds.t2m)
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
            if six.PY3:
                # Test whether the warning is raised if the decoder finds
                # multiple dimensions
                with self.assertWarnsRegex(RuntimeWarning,
                                           'multiple matches'):
                    coords = 'time lat lon lev x y latitude longitude'.split()
                    ds.t2m.attrs.pop('coordinates', None)
                    for dim in 'xytz':
                        getattr(d, dim).update(coords)
                    for coord in set(coords).intersection(ds.coords):
                        ds.coords[coord].attrs.pop('axis', None)
                    getattr(d, func_name)(ds.t2m)
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

    def test_standardization(self):
        """Test the :meth:`psyplot.data.CFDecoder.standardize_dims` method"""
        ds = psyd.open_dataset(os.path.join(bt.test_dir, 'test-t2m-u-v.nc'))
        decoder = psyd.CFDecoder(ds)
        dims = {'time': 1, 'lat': 2, 'lon': 3, 'lev': 4}
        replaced = decoder.standardize_dims(ds.t2m, dims)
        for dim, rep in [('time', 't'), ('lat', 'y'), ('lon', 'x'),
                         ('lev', 'z')]:
            self.assertIn(rep, replaced)
            self.assertEqual(replaced[rep], dims[dim],
                             msg="Wrong value for %s (%s-) dimension" % (
                                 dim, rep))

    def test_idims(self):
        """Test the extraction of the slicers of the dimensions"""
        ds = psyd.open_dataset(os.path.join(bt.test_dir, 'test-t2m-u-v.nc'))
        arr = ds.t2m[1:, 1]
        arr.psy.init_accessor(base=ds)
        dims = arr.psy.idims
        for dim in ['time', 'lev', 'lat', 'lon']:
            self.assertEqual(
                psyd.safe_list(ds[dim][dims[dim]]),
                psyd.safe_list(arr.coords[dim]),
                msg="Slice %s for dimension %s is wrong!" % (dims[dim], dim))

    def test_triangles(self):
        """Test the creation of triangles"""
        ds = psyd.open_dataset(os.path.join(bt.test_dir, 'icon_test.nc'))
        decoder = psyd.CFDecoder(ds)
        var = ds.t2m[0, 0]
        var.attrs.pop('grid_type', None)
        self.assertTrue(decoder.is_triangular(var))
        self.assertTrue(decoder.is_unstructured(var))
        triangles = decoder.get_triangles(var)
        self.assertEqual(len(triangles.triangles), var.size)

        # Test for correct falsification
        ds = psyd.open_dataset(os.path.join(bt.test_dir, 'test-t2m-u-v.nc'))
        decoder = psyd.CFDecoder(ds)
        self.assertFalse(decoder.is_triangular(ds.t2m[0, 0]))
        self.assertFalse(decoder.is_unstructured(ds.t2m[0, 0]))

    def test_is_circumpolar(self):
        """Test whether the is_circumpolar method works"""
        ds = psyd.open_dataset(os.path.join(bt.test_dir,
                                            'circumpolar_test.nc'))
        decoder = psyd.CFDecoder(ds)
        self.assertTrue(decoder.is_circumpolar(ds.t2m))

        # test for correct falsification
        ds = psyd.open_dataset(os.path.join(bt.test_dir, 'icon_test.nc'))
        decoder = psyd.CFDecoder(ds)
        self.assertFalse(decoder.is_circumpolar(ds.t2m))

    def test_get_variable_by_axis(self):
        """Test the :meth:`CFDecoder.get_variable_by_axis` method"""
        ds = psyd.open_dataset(os.path.join(bt.test_dir,
                                            'circumpolar_test.nc'))
        decoder = psyd.CFDecoder(ds)
        arr = ds.t2m
        arr.attrs.pop('coordinates', None)
        for c in ds.coords.values():
            c.attrs.pop('axis', None)
        for dim in ['x', 'y', 'z', 't']:
            self.assertIsNone(decoder.get_variable_by_axis(arr, dim),
                              msg="Accidently found coordinate %s" % dim)

        # test coordinates attribute
        arr.attrs['coordinates'] = 'latitude longitude'
        self.assertEqual(decoder.get_variable_by_axis(arr, 'x').name,
                         'longitude')
        self.assertEqual(decoder.get_variable_by_axis(arr, 'y').name,
                         'latitude')
        self.assertIsNone(decoder.get_variable_by_axis(arr, 'z'))

        # test coordinates attribute but without specifying axis or matching
        # latitude or longitude
        axes = {'lev': 'z', 'time': 't', 'x': 'x', 'y': 'y'}
        arr.attrs['coordinates'] = 'time lev y x'
        for name, axis in axes.items():
            self.assertEqual(
                decoder.get_variable_by_axis(arr, axis).name, name)

        # test with specified axis attribute
        arr.attrs['coordinates'] = 'time lev longitude latitude'
        axes = {'lev': 'Z', 'time': 'T', 'latitude': 'X', 'longitude': 'Y'}
        for name, axis in axes.items():
            ds.coords[name].attrs['axis'] = axis
        for name, axis in axes.items():
            self.assertEqual(
                decoder.get_variable_by_axis(arr, axis.lower()).name, name)

        # close the dataset
        ds.close()


class TestTempBool(unittest.TestCase):
    """Test the :class:`psyplot.data._TempBool` class"""

    def test_descriptor(self):
        """Test the descriptor functionality"""

        class Test(object):

            test = psyd._temp_bool_prop('test')

        t = Test()

        self.assertFalse(t.test)
        with t.test:
            self.assertTrue(t.test)

        t.test = True
        self.assertTrue(t.test)
        with t.test:
            self.assertTrue(t.test)

        del t.test
        self.assertFalse(t.test)


class TestArrayList(unittest.TestCase):
    """Test the :class:`psyplot.data.ArrayList` class"""

    def test_setup_coords(self):
        """Set the :func:`psyplot.data.setup_coords` function"""
        coords = {'first': [1, 2]}
        self.assertEqual(psyd.setup_coords(second=3, **coords),
                         {'arr0': {'first': 1, 'second': 3},
                          'arr1': {'first': 2, 'second': 3}})
        self.assertEqual(psyd.setup_coords(dims=coords, second=3),
                         {'arr0': {'first': 1, 'second': 3},
                          'arr1': {'first': 2, 'second': 3}})
        coords['third'] = [1, 2, 3]
        # test sorting
        ret = psyd.setup_coords(arr_names='test{}', second=3,
                                sort=['third', 'first'], **coords)
        self.assertEqual(ret, {
            'test0': {'third': 1, 'first': 1, 'second': 3},
            'test1': {'third': 1, 'first': 2, 'second': 3},
            'test2': {'third': 2, 'first': 1, 'second': 3},
            'test3': {'third': 2, 'first': 2, 'second': 3},
            'test4': {'third': 3, 'first': 1, 'second': 3},
            'test5': {'third': 3, 'first': 2, 'second': 3}})

    @property
    def _filter_test_ds(self):
        return xr.Dataset(
            {'arr1': xr.Variable(('ydim', 'xdim'), np.zeros((4, 4)),
                                 attrs={'test': 1, 'test2': 1}),
             'arr2': xr.Variable(('xdim', ), np.zeros(4), attrs={'test': 2,
                                                                 'test2': 2}),
             'arr3': xr.Variable(('xdim', ), np.zeros(4), attrs={'test': 3,
                                                                 'test2': 3})},
            {'ydim': xr.Variable(('ydim', ), np.arange(1, 5))})

    def test_filter_1_name(self):
        """Test the filtering of the ArrayList"""
        ds = self._filter_test_ds
        l = psyd.ArrayList.from_dataset(ds, ydim=0)
        l.extend(psyd.ArrayList.from_dataset(ds, ydim=1, name='arr1'),
                 new_name=True)
        # filter by name
        self.assertEqual([arr.name for arr in l(name='arr2')],
                         ['arr2'])
        self.assertEqual([arr.name for arr in l(name=['arr2', 'arr3'])],
                         ['arr2', 'arr3'])
        self.assertEqual(
            [arr.psy.arr_name for arr in l(
                 arr_name=lambda name: name == 'arr1')], ['arr1'])

    def test_filter_2_arr_name(self):
        """Test the filtering of the ArrayList"""
        ds = self._filter_test_ds
        l = psyd.ArrayList.from_dataset(ds, ydim=0)
        l.extend(psyd.ArrayList.from_dataset(ds, ydim=1, name='arr1'),
                 new_name=True)
        # fillter by array name
        self.assertEqual([arr.psy.arr_name for arr in l(arr_name='arr1')],
                         ['arr1'])
        self.assertEqual([arr.psy.arr_name for arr in l(arr_name=['arr1',
                                                                  'arr2'])],
                         ['arr1', 'arr2'])
        self.assertEqual(
            [arr.psy.arr_name for arr in l(
                 name=lambda name: name == 'arr2')], ['arr1'])

    def test_filter_3_attribute(self):
        """Test the filtering of the ArrayList"""
        ds = self._filter_test_ds
        l = psyd.ArrayList.from_dataset(ds, ydim=0)
        l.extend(psyd.ArrayList.from_dataset(ds, ydim=1, name='arr1'),
                 new_name=True)
        # filter by attribute
        self.assertEqual([arr.name for arr in l(test=2)], ['arr2'])
        self.assertEqual([arr.name for arr in l(test=[2, 3])],
                         ['arr2', 'arr3'])
        self.assertEqual([arr.name for arr in l(test=[1, 2], test2=2)],
                         ['arr2'])
        self.assertEqual(
            [arr.psy.arr_name for arr in l(test=lambda val: val == 2)],
            ['arr1'])

    def test_filter_4_coord(self):
        """Test the filtering of the ArrayList"""
        ds = self._filter_test_ds
        l = psyd.ArrayList.from_dataset(ds, ydim=0)
        l.extend(psyd.ArrayList.from_dataset(ds, ydim=1, name='arr1'),
                 new_name=True)
        # filter by coordinate
        self.assertEqual([arr.psy.arr_name for arr in l(y=0)], ['arr0'])
        self.assertEqual([arr.psy.arr_name for arr in l(y=1)], ['arr3'])
        self.assertEqual([arr.psy.arr_name for arr in l(y=1, method='sel')],
                         ['arr0'])
        self.assertEqual(
            [arr.psy.arr_name for arr in l(y=lambda val: val == 0)], ['arr0'])

    def test_filter_5_mixed(self):
        """Test the filtering of the ArrayList"""
        ds = self._filter_test_ds
        l = psyd.ArrayList.from_dataset(ds, ydim=0)
        l.extend(psyd.ArrayList.from_dataset(ds, ydim=1, name='arr1'),
                 new_name=True)
        # mix criteria
        self.assertEqual(
            [arr.psy.arr_name for arr in l(arr_name=['arr0', 'arr1'], test=1)],
            ['arr0'])

    def _from_dataset_test_variables(self):
        variables = {
             # 3d-variable
             'v0': xr.Variable(('time', 'ydim', 'xdim'), np.zeros((4, 4, 4))),
             # 2d-variable with time and x
             'v1': xr.Variable(('time', 'xdim', ), np.zeros((4, 4))),
             # 2d-variable with y and x
             'v2': xr.Variable(('ydim', 'xdim', ), np.zeros((4, 4))),
             # 1d-variable
             'v3': xr.Variable(('xdim', ), np.zeros(4))}
        coords = {
            'ydim': xr.Variable(('ydim', ), np.arange(1, 5)),
            'time': xr.Variable(
                ('time', ),
                pd.date_range('1999-01-01', '1999-05-01', freq='M').values)}
        return variables, coords

    def test_from_dataset_1_basic(self):
        """test creation without any additional information"""
        variables, coords = self._from_dataset_test_variables()
        ds = xr.Dataset(variables, coords)
        l = psyd.ArrayList.from_dataset(ds)
        self.assertEqual(len(l), 4)
        self.assertEqual(set(l.names), set(variables))
        for arr in l:
            self.assertEqual(arr.dims, variables[arr.name].dims,
                             msg="Wrong dimensions for variable " + arr.name)
            self.assertEqual(arr.shape, variables[arr.name].shape,
                             msg="Wrong shape for variable " + arr.name)

    def test_from_dataset_2_name(self):
        """Test the from_dataset creation method with selected names"""
        variables, coords = self._from_dataset_test_variables()
        ds = xr.Dataset(variables, coords)
        l = psyd.ArrayList.from_dataset(ds, name="v2")
        self.assertEqual(len(l), 1)
        self.assertEqual(set(l.names), {"v2"})
        for arr in l:
            self.assertEqual(arr.dims, variables[arr.name].dims,
                             msg="Wrong dimensions for variable " + arr.name)
            self.assertEqual(arr.shape, variables[arr.name].shape,
                             msg="Wrong shape for variable " + arr.name)

    def test_from_dataset_3_simple_selection(self):
        """Test the from_dataset creation method with x- and t-selection"""
        variables, coords = self._from_dataset_test_variables()
        ds = xr.Dataset(variables, coords)
        l = psyd.ArrayList.from_dataset(ds, x=0, t=0)
        self.assertEqual(len(l), 4)
        self.assertEqual(set(l.names), set(variables))
        for arr in l:
            self.assertEqual(arr.xdim.ndim, 0,
                             msg="Wrong x dimension for " + arr.name)
            if 'time' in arr.dims:
                self.assertEqual(arr.time, coords['time'],
                                 msg="Wrong time dimension for " + arr.name)

    def test_from_dataset_4_exact_selection(self):
        """Test the from_dataset creation method with selected names"""
        variables, coords = self._from_dataset_test_variables()
        ds = xr.Dataset(variables, coords)
        l = psyd.ArrayList.from_dataset(ds, ydim=2, method=None,
                                        name=['v0', 'v2'])
        self.assertEqual(len(l), 2)
        self.assertEqual(set(l.names), {'v0', 'v2'})
        for arr in l:
            self.assertEqual(arr.ydim, 2,
                             msg="Wrong ydim slice for " + arr.name)

    def test_from_dataset_5_exact_array_selection(self):
        """Test the from_dataset creation method with selected names"""
        variables, coords = self._from_dataset_test_variables()
        ds = xr.Dataset(variables, coords)
        l = psyd.ArrayList.from_dataset(ds, ydim=[[2, 3]], method=None,
                                        name=['v0', 'v2'])
        self.assertEqual(len(l), 2)
        self.assertEqual(set(l.names), {'v0', 'v2'})
        for arr in l:
            self.assertEqual(arr.ydim.values.tolist(), [2, 3],
                             msg="Wrong ydim slice for " + arr.name)

    def test_from_dataset_6_nearest_selection(self):
        """Test the from_dataset creation method with selected names"""
        variables, coords = self._from_dataset_test_variables()
        ds = xr.Dataset(variables, coords)
        l = psyd.ArrayList.from_dataset(ds, ydim=1.7, method='nearest',
                                        name=['v0', 'v2'])
        self.assertEqual(len(l), 2)
        self.assertEqual(set(l.names), {'v0', 'v2'})
        for arr in l:
            self.assertEqual(arr.ydim, 2,
                             msg="Wrong ydim slice for " + arr.name)

    def test_from_dataset_7_time_selection(self):
        """Test the from_dataset creation method with selected names"""
        variables, coords = self._from_dataset_test_variables()
        ds = xr.Dataset(variables, coords)
        l = psyd.ArrayList.from_dataset(ds, t='1999-02-28', method=None,
                                        name=['v0', 'v1'])
        self.assertEqual(len(l), 2)
        self.assertEqual(set(l.names), {'v0', 'v1'})
        for arr in l:
            self.assertEqual(arr.time, coords['time'][1],
                             msg="Wrong time slice for " + arr.name)

    def test_from_dataset_8_time_array_selection(self):
        """Test the from_dataset creation method with selected names"""
        variables, coords = self._from_dataset_test_variables()
        ds = xr.Dataset(variables, coords)
        # test with array of time
        l = psyd.ArrayList.from_dataset(ds, t=[coords['time'][1:3]],
                                        method=None, name=['v0', 'v1'])
        self.assertEqual(len(l), 2)
        self.assertEqual(set(l.names), {'v0', 'v1'})
        for arr in l:
            self.assertEqual(arr.time.values.tolist(),
                             coords['time'][1:3].values.tolist(),
                             msg="Wrong time slice for " + arr.name)

    def test_from_dataset_9_nearest_time_selection(self):
        """Test the from_dataset creation method with selected names"""
        variables, coords = self._from_dataset_test_variables()
        ds = xr.Dataset(variables, coords)
        l = psyd.ArrayList.from_dataset(ds, t='1999-02-20', method='nearest',
                                        name=['v0', 'v1'])
        self.assertEqual(len(l), 2)
        self.assertEqual(set(l.names), {'v0', 'v1'})
        for arr in l:
            self.assertEqual(arr.time, coords['time'][1],
                             msg="Wrong time slice for " + arr.name)


if __name__ == '__main__':
    unittest.main()
