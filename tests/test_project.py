"""Test module of the :mod:`psyplot.project` module"""
import os
import os.path as osp
import six
import unittest
from itertools import chain
import _base_testing as bt
import test_data as td
import test_plotter as tp
import xarray as xr
import psyplot.data as psyd
import psyplot.project as psy
import matplotlib.pyplot as plt

from test_plotter import TestPlotter, SimpleFmt


def get_file(fname):
    return os.path.join(bt.test_dir, fname)


class TestProject(td.TestArrayList):
    """Testclass for the :class:`psyplot.project.Project` class"""

    def setUp(self):
        psy.close('all')
        plt.close('all')

    def tearDown(self):
        for identifier in list(psy.registered_plotters):
            psy.unregister_plotter(identifier)
        psy.close('all')
        plt.close('all')
        tp.results.clear()

    def test_save_and_load_01_simple(self):
        """Test the saving and loading of a Project"""
        psy.register_plotter('test_plotter', import_plotter=True,
                             module='test_plotter', plotter_name='TestPlotter')
        ds = psy.open_dataset(bt.get_file('test-t2m-u-v.nc'))
        plt.close('all')
        sp = psy.plot.test_plotter(ds, name=['t2m', 'u'], x=0, y=4,
                                   ax=(2, 2, 1), fmt1='test')
        self.assertEqual(len(sp), 2)
        self.assertEqual(sp[0].psy.ax.get_figure().number, 1)
        self.assertEqual(sp[0].psy.ax.rowNum, 0)
        self.assertEqual(sp[0].psy.ax.colNum, 0)
        self.assertEqual(sp[0].psy.ax.numCols, 2)
        self.assertEqual(sp[0].psy.ax.numRows, 2)
        self.assertEqual(sp[1].psy.ax.get_figure().number, 2)
        self.assertEqual(sp[1].psy.ax.rowNum, 0)
        self.assertEqual(sp[1].psy.ax.colNum, 0)
        self.assertEqual(sp[1].psy.ax.numCols, 2)
        self.assertEqual(sp[1].psy.ax.numRows, 2)
        arr_names = sp.arr_names
        self.assertEqual(tp.results[arr_names[0] + '.fmt1'], 'test')
        self.assertEqual(tp.results[arr_names[1] + '.fmt1'], 'test')
        fname = 'test.pkl'
        sp.save_project(fname)
        psy.close()
        tp.results.clear()
        sp = psy.Project.load_project(fname)
        self.assertEqual(len(sp), 2)
        self.assertEqual(sp[0].psy.ax.get_figure().number, 1)
        self.assertEqual(sp[0].psy.ax.rowNum, 0)
        self.assertEqual(sp[0].psy.ax.colNum, 0)
        self.assertEqual(sp[0].psy.ax.numCols, 2)
        self.assertEqual(sp[0].psy.ax.numRows, 2)
        self.assertEqual(sp[1].psy.ax.get_figure().number, 2)
        self.assertEqual(sp[1].psy.ax.rowNum, 0)
        self.assertEqual(sp[1].psy.ax.colNum, 0)
        self.assertEqual(sp[1].psy.ax.numCols, 2)
        self.assertEqual(sp[1].psy.ax.numRows, 2)

        psy.close()
        psy.unregister_plotter('test_plotter')
        tp.results.clear()

        if osp.exists(fname):
            os.remove(fname)

    def test_save_and_load_02_alternative_axes(self):
        """Test the saving and loading of a Project providing alternative axes
        """
        psy.register_plotter('test_plotter', import_plotter=True,
                             module='test_plotter', plotter_name='TestPlotter')
        ds = psy.open_dataset(bt.get_file('test-t2m-u-v.nc'))
        plt.close('all')
        sp = psy.plot.test_plotter(ds, name=['t2m', 'u'], x=0, y=4,
                                   ax=(2, 2, 1), fmt1='test')
        self.assertEqual(len(sp), 2)
        self.assertEqual(sp[0].psy.ax.get_figure().number, 1)
        self.assertEqual(sp[0].psy.ax.rowNum, 0)
        self.assertEqual(sp[0].psy.ax.colNum, 0)
        self.assertEqual(sp[0].psy.ax.numCols, 2)
        self.assertEqual(sp[0].psy.ax.numRows, 2)
        self.assertEqual(sp[1].psy.ax.get_figure().number, 2)
        self.assertEqual(sp[1].psy.ax.rowNum, 0)
        self.assertEqual(sp[1].psy.ax.colNum, 0)
        self.assertEqual(sp[1].psy.ax.numCols, 2)
        self.assertEqual(sp[1].psy.ax.numRows, 2)
        arr_names = sp.arr_names
        self.assertEqual(tp.results[arr_names[0] + '.fmt1'], 'test')
        self.assertEqual(tp.results[arr_names[1] + '.fmt1'], 'test')
        fname = 'test.pkl'
        sp.save_project(fname)
        psy.close()
        tp.results.clear()
        fig, axes = plt.subplots(1, 2)
        sp = psy.Project.load_project(fname, alternative_axes=axes.ravel())
        self.assertEqual(len(sp), 2)
        self.assertEqual(sp[0].psy.ax.get_figure().number, 1)
        self.assertEqual(sp[0].psy.ax.rowNum, 0)
        self.assertEqual(sp[0].psy.ax.colNum, 0)
        self.assertEqual(sp[0].psy.ax.numCols, 2)
        self.assertEqual(sp[0].psy.ax.numRows, 1)
        self.assertEqual(sp[1].psy.ax.get_figure().number, 1)
        self.assertEqual(sp[1].psy.ax.rowNum, 0)
        self.assertEqual(sp[1].psy.ax.colNum, 1)
        self.assertEqual(sp[1].psy.ax.numCols, 2)
        self.assertEqual(sp[1].psy.ax.numRows, 1)

        psy.close()
        psy.unregister_plotter('test_plotter')
        tp.results.clear()

        if osp.exists(fname):
            os.remove(fname)

    def test_save_and_load_03_alternative_ds(self):
        """Test the saving and loading of a Project providing alternative axes
        """
        psy.register_plotter('test_plotter', import_plotter=True,
                             module='test_plotter', plotter_name='TestPlotter')
        ds = psy.open_dataset(bt.get_file('test-t2m-u-v.nc'))
        plt.close('all')
        sp = psy.plot.test_plotter(ds, name=['t2m', 'u'], x=0, y=4,
                                   ax=(2, 2, 1), fmt1='test')
        self.assertEqual(len(sp), 2)
        self.assertEqual(sp[0].psy.ax.get_figure().number, 1)
        self.assertEqual(sp[0].psy.ax.rowNum, 0)
        self.assertEqual(sp[0].psy.ax.colNum, 0)
        self.assertEqual(sp[0].psy.ax.numCols, 2)
        self.assertEqual(sp[0].psy.ax.numRows, 2)
        self.assertEqual(sp[1].psy.ax.get_figure().number, 2)
        self.assertEqual(sp[1].psy.ax.rowNum, 0)
        self.assertEqual(sp[1].psy.ax.colNum, 0)
        self.assertEqual(sp[1].psy.ax.numCols, 2)
        self.assertEqual(sp[1].psy.ax.numRows, 2)
        arr_names = sp.arr_names
        self.assertEqual(tp.results[arr_names[0] + '.fmt1'], 'test')
        self.assertEqual(tp.results[arr_names[1] + '.fmt1'], 'test')
        fname = 'test.pkl'
        sp.save_project(fname)
        psy.close()
        tp.results.clear()
        fig, axes = plt.subplots(1, 2)
        ds = psy.open_dataset(bt.get_file('circumpolar_test.nc'))
        sp = psy.Project.load_project(fname, datasets=[ds])
        self.assertEqual(len(sp), 2)
        self.assertEqual(sp[0].psy.ax.get_figure().number, 1)
        self.assertEqual(sp[0].psy.ax.rowNum, 0)
        self.assertEqual(sp[0].psy.ax.colNum, 0)
        self.assertEqual(sp[0].psy.ax.numCols, 2)
        self.assertEqual(sp[0].psy.ax.numRows, 2)
        self.assertEqual(sp[1].psy.ax.get_figure().number, 2)
        self.assertEqual(sp[1].psy.ax.rowNum, 0)
        self.assertEqual(sp[1].psy.ax.colNum, 0)
        self.assertEqual(sp[1].psy.ax.numCols, 2)
        self.assertEqual(sp[1].psy.ax.numRows, 2)
        self.assertIs(sp[0].psy.base, ds)
        self.assertIs(sp[1].psy.base, ds)

        psy.close()
        psy.unregister_plotter('test_plotter')
        tp.results.clear()

        if osp.exists(fname):
            os.remove(fname)

    def test_save_and_load_04_alternative_fname(self):
        """Test the saving and loading of a Project providing alternative axes
        """
        psy.register_plotter('test_plotter', import_plotter=True,
                             module='test_plotter', plotter_name='TestPlotter')
        ds = psy.open_dataset(bt.get_file('test-t2m-u-v.nc'))
        plt.close('all')
        sp = psy.plot.test_plotter(ds, name=['t2m', 'u'], x=0, y=4,
                                   ax=(2, 2, 1), fmt1='test')
        self.assertEqual(len(sp), 2)
        self.assertEqual(sp[0].psy.ax.get_figure().number, 1)
        self.assertEqual(sp[0].psy.ax.rowNum, 0)
        self.assertEqual(sp[0].psy.ax.colNum, 0)
        self.assertEqual(sp[0].psy.ax.numCols, 2)
        self.assertEqual(sp[0].psy.ax.numRows, 2)
        self.assertEqual(sp[1].psy.ax.get_figure().number, 2)
        self.assertEqual(sp[1].psy.ax.rowNum, 0)
        self.assertEqual(sp[1].psy.ax.colNum, 0)
        self.assertEqual(sp[1].psy.ax.numCols, 2)
        self.assertEqual(sp[1].psy.ax.numRows, 2)
        arr_names = sp.arr_names
        self.assertEqual(tp.results[arr_names[0] + '.fmt1'], 'test')
        self.assertEqual(tp.results[arr_names[1] + '.fmt1'], 'test')
        fname = 'test.pkl'
        sp.save_project(fname)
        psy.close()
        tp.results.clear()
        fig, axes = plt.subplots(1, 2)
        sp = psy.Project.load_project(
            fname, alternative_paths=[bt.get_file('circumpolar_test.nc')])
        self.assertEqual(len(sp), 2)
        self.assertEqual(sp[0].psy.ax.get_figure().number, 1)
        self.assertEqual(sp[0].psy.ax.rowNum, 0)
        self.assertEqual(sp[0].psy.ax.colNum, 0)
        self.assertEqual(sp[0].psy.ax.numCols, 2)
        self.assertEqual(sp[0].psy.ax.numRows, 2)
        self.assertEqual(sp[1].psy.ax.get_figure().number, 2)
        self.assertEqual(sp[1].psy.ax.rowNum, 0)
        self.assertEqual(sp[1].psy.ax.colNum, 0)
        self.assertEqual(sp[1].psy.ax.numCols, 2)
        self.assertEqual(sp[1].psy.ax.numRows, 2)
        self.assertEqual(psyd.get_filename_ds(sp[0].psy.base)[0],
                         bt.get_file('circumpolar_test.nc'))
        self.assertEqual(psyd.get_filename_ds(sp[1].psy.base)[0],
                         bt.get_file('circumpolar_test.nc'))

        psy.close()
        psy.unregister_plotter('test_plotter')
        tp.results.clear()

        if osp.exists(fname):
            os.remove(fname)

    def test_keys(self):
        """Test the :meth:`psyplot.project.Project.keys` method"""
        import test_plotter as tp
        import psyplot.plotter as psyp
        psy.register_plotter('test_plotter', import_plotter=True,
                             module='test_plotter', plotter_name='TestPlotter')

        class TestPlotter2(tp.TestPlotter):
            fmt2 = None

        psy.register_plotter('test_plotter2', module='something',
                             plotter_name='anyway', plotter_cls=TestPlotter2)
        variables, coords = self._from_dataset_test_variables
        ds = xr.Dataset(variables, coords)
        sp1 = psy.plot.test_plotter(ds, name='v0')
        # add a second project without a fmt2 formatoption
        sp2 = psy.plot.test_plotter2(ds, name='v1')
        mp = sp1 + sp2
        self.assertEqual(sp1.keys(func=str),
                         '+------+------+------+\n'
                         '| fmt1 | fmt2 | fmt3 |\n'
                         '+------+------+------+')
        self.assertEqual(mp.keys(func=str),
                         '+------+------+\n'
                         '| fmt1 | fmt3 |\n'
                         '+------+------+')
        title = psyp.groups['labels']
        self.assertEqual(sp1.keys(func=str, grouped=True),
                         '*' * len(title) + '\n' +
                         title + '\n' +
                         '*' * len(title) + '\n'
                         '+------+------+\n'
                         '| fmt1 | fmt2 |\n'
                         '+------+------+\n'
                         '\n'
                         '*********\n'
                         'something\n'
                         '*********\n'
                         '+------+\n'
                         '| fmt3 |\n'
                         '+------+')
        self.assertEqual(mp.keys(func=str, grouped=True),
                         '*' * len(title) + '\n' +
                         title + '\n' +
                         '*' * len(title) + '\n'
                         '+------+\n'
                         '| fmt1 |\n'
                         '+------+\n'
                         '\n'
                         '*********\n'
                         'something\n'
                         '*********\n'
                         '+------+\n'
                         '| fmt3 |\n'
                         '+------+')
        self.assertEqual(sp1.keys(['fmt1', 'something'], func=str),
                         '+------+------+\n'
                         '| fmt1 | fmt3 |\n'
                         '+------+------+')
        if six.PY3:
            with self.assertWarnsRegex(UserWarning,
                                       '(?i)unknown formatoption keyword'):
                self.assertEqual(
                    sp1.keys(['fmt1', 'wrong', 'something'], func=str),
                    '+------+------+\n'
                    '| fmt1 | fmt3 |\n'
                    '+------+------+')

    def test_docs(self):
        """Test the :meth:`psyplot.project.Project.docs` method"""
        import test_plotter as tp
        import psyplot.plotter as psyp
        psy.register_plotter('test_plotter', import_plotter=True,
                             module='test_plotter', plotter_name='TestPlotter')

        class TestPlotter2(tp.TestPlotter):
            fmt2 = None

        psy.register_plotter('test_plotter2', module='something',
                             plotter_name='anyway', plotter_cls=TestPlotter2)
        variables, coords = self._from_dataset_test_variables
        ds = xr.Dataset(variables, coords)
        sp1 = psy.plot.test_plotter(ds, name='v0')
        # add a second project without a fmt2 formatoption
        sp2 = psy.plot.test_plotter2(ds, name='v1')
        mp = sp1 + sp2
        self.assertEqual(sp1.docs(func=str), '\n'.join([
            'fmt1', '====', tp.SimpleFmt.__doc__, '',
            'fmt2', '====', tp.SimpleFmt2.__doc__, '',
            'fmt3', '====', tp.SimpleFmt3.__doc__, '']))
        # test summed project
        self.assertEqual(mp.docs(func=str), '\n'.join([
            'fmt1', '====', tp.SimpleFmt.__doc__, '',
            'fmt3', '====', tp.SimpleFmt3.__doc__, '']))
        title = psyp.groups['labels']
        self.assertEqual(sp1.docs(func=str, grouped=True), '\n'.join([
            '*' * len(title),
            title,
            '*' * len(title),
            'fmt1', '====', tp.SimpleFmt.__doc__, '',
            'fmt2', '====', tp.SimpleFmt2.__doc__, '', '',
            '*********',
            'something',
            '*********',
            'fmt3', '====', tp.SimpleFmt3.__doc__]))
        # test summed project
        self.assertEqual(mp.docs(func=str, grouped=True), '\n'.join([
            '*' * len(title),
            title,
            '*' * len(title),
            'fmt1', '====', tp.SimpleFmt.__doc__, '', '',
            '*********',
            'something',
            '*********',
            'fmt3', '====', tp.SimpleFmt3.__doc__]))

    def test_summaries(self):
        """Test the :meth:`psyplot.project.Project.summaries` method"""
        import test_plotter as tp
        import psyplot.plotter as psyp
        psy.register_plotter('test_plotter', import_plotter=True,
                             module='test_plotter', plotter_name='TestPlotter')

        class TestPlotter2(tp.TestPlotter):
            fmt2 = None

        psy.register_plotter('test_plotter2', module='something',
                             plotter_name='anyway', plotter_cls=TestPlotter2)
        variables, coords = self._from_dataset_test_variables
        ds = xr.Dataset(variables, coords)
        sp1 = psy.plot.test_plotter(ds, name='v0')
        # add a second project without a fmt2 formatoption
        sp2 = psy.plot.test_plotter2(ds, name='v1')
        mp = sp1 + sp2
        self.assertEqual(sp1.summaries(func=str), '\n'.join([
            'fmt1', tp.indent(tp.SimpleFmt.__doc__.splitlines()[0], '    '),
            'fmt2', tp.indent(tp.SimpleFmt2.__doc__.splitlines()[0], '    '),
            'fmt3', tp.indent(tp.SimpleFmt3.__doc__.splitlines()[0], '    ')]))
        # test summed project
        self.assertEqual(mp.summaries(func=str), '\n'.join([
            'fmt1', tp.indent(tp.SimpleFmt.__doc__.splitlines()[0], '    '),
            'fmt3', tp.indent(tp.SimpleFmt3.__doc__.splitlines()[0], '    ')]))
        title = psyp.groups['labels']
        self.assertEqual(sp1.summaries(func=str, grouped=True), '\n'.join([
            '*' * len(title),
            title,
            '*' * len(title),
            'fmt1', tp.indent(tp.SimpleFmt.__doc__.splitlines()[0], '    '),
            'fmt2', tp.indent(tp.SimpleFmt2.__doc__.splitlines()[0], '    '),
            '',
            '*********',
            'something',
            '*********',
            'fmt3', tp.indent(tp.SimpleFmt3.__doc__.splitlines()[0], '    ')]
            ))
        # test summed project
        self.assertEqual(mp.summaries(func=str, grouped=True), '\n'.join([
            '*' * len(title),
            title,
            '*' * len(title),
            'fmt1', tp.indent(tp.SimpleFmt.__doc__.splitlines()[0], '    '),
            '',
            '*********',
            'something',
            '*********',
            'fmt3', tp.indent(tp.SimpleFmt3.__doc__.splitlines()[0], '    ')]
            ))

    def test_figs(self):
        """Test the :attr:`psyplot.project.Project.figs` attribute"""
        psy.register_plotter('test_plotter', import_plotter=True,
                             module='test_plotter', plotter_name='TestPlotter')
        ds = psy.open_dataset(bt.get_file('test-t2m-u-v.nc'))
        sp = psy.plot.test_plotter(ds, name='t2m', time=[1, 2])
        self.assertEqual(sp[0].psy.ax.figure.number, 1)
        self.assertEqual(sp[1].psy.ax.figure.number, 2)
        figs = sp.figs
        self.assertIn(sp[0].psy.ax.figure, figs)
        self.assertIs(figs[sp[0].psy.ax.figure][0], sp[0])
        self.assertIn(sp[1].psy.ax.figure, figs)
        self.assertIs(figs[sp[1].psy.ax.figure][0], sp[1])

    def test_axes(self):
        """Test the :attr:`psyplot.project.Project.axes` attribute"""
        psy.register_plotter('test_plotter', import_plotter=True,
                             module='test_plotter', plotter_name='TestPlotter')
        ds = psy.open_dataset(bt.get_file('test-t2m-u-v.nc'))
        sp = psy.plot.test_plotter(ds, name='t2m', time=[1, 2])
        self.assertIsNot(sp[0].psy.ax, sp[1].psy.ax)
        axes = sp.axes
        self.assertIn(sp[0].psy.ax, axes)
        self.assertIs(axes[sp[0].psy.ax][0], sp[0])
        self.assertIn(sp[1].psy.ax, axes)
        self.assertIs(axes[sp[1].psy.ax][0], sp[1])


class TestPlotterInterface(unittest.TestCase):

    list_class = psy.Project

    def test_plotter_registration(self):
        """Test the registration of a plotter"""
        psy.register_plotter('test_plotter',
                             import_plotter=True, module='test_plotter',
                             plotter_name='TestPlotter')
        self.assertTrue(hasattr(psy.plot, 'test_plotter'))
        self.assertIs(psy.plot.test_plotter.plotter_cls, TestPlotter)
        psy.plot.test_plotter.print_func = str
        self.assertEqual(psy.plot.test_plotter.fmt1, SimpleFmt.__doc__)
        psy.plot.test_plotter.print_func = None
        # test the warning
        if not six.PY2:
            with self.assertWarnsRegex(UserWarning, "not_existent_module"):
                psy.register_plotter('something', "not_existent_module",
                                     'not_important', import_plotter=True)
        psy.unregister_plotter('test_plotter')
        self.assertFalse(hasattr(psy.Project, 'test_plotter'))
        self.assertFalse(hasattr(psy.plot, 'test_plotter'))

    def test_plot_creation_01_array(self):
        """Test the plot creation with a plotter that takes one array"""
        psy.register_plotter('test_plotter',
                             import_plotter=True, module='test_plotter',
                             plotter_name='TestPlotter')
        ds = psy.open_dataset(bt.get_file('test-t2m-u-v.nc'))
        sp = psy.plot.test_plotter(ds, name='t2m')
        self.assertEqual(len(sp), 1)
        self.assertEqual(sp[0].name, 't2m')
        self.assertEqual(sp[0].shape, ds.t2m.shape)
        self.assertEqual(sp[0].values.tolist(), ds.t2m.values.tolist())
        psy.close()
        psy.unregister_plotter('test_plotter')

    def test_plot_creation_02_array_default_dims(self):
        # add a default value for the y dimension
        psy.register_plotter('test_plotter',
                             import_plotter=True, module='test_plotter',
                             plotter_name='TestPlotter',
                             default_dims={'y': 0})
        ds = psy.open_dataset(bt.get_file('test-t2m-u-v.nc'))
        sp = psy.plot.test_plotter(ds, name='t2m')
        self.assertEqual(len(sp), 1)
        self.assertEqual(sp[0].name, 't2m')
        self.assertEqual(sp[0].shape, ds.t2m.isel(lat=0).shape)
        self.assertEqual(sp[0].values.tolist(),
                         ds.t2m.isel(lat=0).values.tolist())
        psy.close()
        psy.unregister_plotter('test_plotter')

    def test_plot_creation_03_2arrays(self):
        # try multiple names and dimension
        psy.register_plotter('test_plotter',
                             import_plotter=True, module='test_plotter',
                             plotter_name='TestPlotter',
                             default_dims={'y': 0})
        ds = psy.open_dataset(bt.get_file('test-t2m-u-v.nc'))
        sp = psy.plot.test_plotter(ds, name=['t2m', 'u'], x=slice(3, 5))
        self.assertEqual(len(sp), 2)
        self.assertEqual(sp[0].name, 't2m')
        self.assertEqual(sp[1].name, 'u')
        self.assertEqual(sp[0].shape,
                         ds.t2m.isel(lat=0, lon=slice(3, 5)).shape)
        self.assertEqual(sp[1].shape,
                         ds.u.isel(lat=0, lon=slice(3, 5)).shape)
        self.assertEqual(sp[0].values.tolist(),
                         ds.t2m.isel(lat=0, lon=slice(3, 5)).values.tolist())
        self.assertEqual(sp[1].values.tolist(),
                         ds.u.isel(lat=0, lon=slice(3, 5)).values.tolist())
        psy.close()
        psy.unregister_plotter('test_plotter')

    def test_plot_creation_04_2variables(self):
        # test with array out of 2 variables
        psy.register_plotter('test_plotter',
                             import_plotter=True, module='test_plotter',
                             plotter_name='TestPlotter',
                             default_dims={'y': 0})
        ds = psy.open_dataset(bt.get_file('test-t2m-u-v.nc'))
        sp = psy.plot.test_plotter(ds, name=[['u', 'v']], x=slice(3, 5))
        self.assertEqual(len(sp), 1)
        self.assertIn('variable', sp[0].dims)
        self.assertEqual(sp[0].coords['variable'].values.tolist(), ['u', 'v'])
        self.assertEqual(list(sp[0].shape),
                         [2] + list(ds.t2m.isel(lat=0, lon=slice(3, 5)).shape))
        self.assertEqual(sp[0].values.tolist(),
                         ds[['u', 'v']].to_array().isel(
                             lat=0, lon=slice(3, 5)).values.tolist())
        psy.close()
        psy.unregister_plotter('test_plotter')

    def test_plot_creation_05_array_and_2variables(self):
        # test a combination of them
        # psyplot.project.Project([
        #     arr0: 2-dim DataArray of t2m, with
        #         (time, lev)=(5, 4), lon=1.875, lat=88.5721685,
        #     arr1: 2-dim DataArray of t2m, with
        #         (time, lev)=(5, 4), lon=3.75, lat=88.5721685,
        #     arr2: 3-dim DataArray of u, v, with
        #         (variable, time, lev)=(2, 5, 4), lat=88.5721685, lon=1.875,
        #     arr3: 3-dim DataArray of u, v, with
        #         (variable, time, lev)=(2, 5, 4), lat=88.5721685, lon=3.75])
        psy.register_plotter('test_plotter',
                             import_plotter=True, module='test_plotter',
                             plotter_name='TestPlotter',
                             default_dims={'y': 0})
        ds = psy.open_dataset(bt.get_file('test-t2m-u-v.nc'))
        sp = psy.plot.test_plotter(ds, name=['t2m', ['u', 'v']], x=[1, 2])
        self.assertEqual(len(sp), 4, msg=str(sp))
        self.assertEqual(sp[0].shape, ds.t2m.isel(lat=0, lon=1).shape)
        self.assertEqual(sp[1].shape, ds.t2m.isel(lat=0, lon=2).shape)
        self.assertEqual(list(sp[2].shape),
                         [2] + list(ds.u.isel(lat=0, lon=1).shape))
        self.assertEqual(list(sp[2].shape),
                         [2] + list(ds.u.isel(lat=0, lon=2).shape))
        self.assertEqual(sp[0].values.tolist(),
                         ds.t2m.isel(lat=0, lon=1).values.tolist())
        self.assertEqual(sp[1].values.tolist(),
                         ds.t2m.isel(lat=0, lon=2).values.tolist())
        self.assertEqual(sp[2].values.tolist(),
                         ds[['u', 'v']].isel(
                             lat=0, lon=1).to_array().values.tolist())
        self.assertEqual(sp[3].values.tolist(),
                         ds[['u', 'v']].isel(
                             lat=0, lon=2).to_array().values.tolist())
        psy.close()
        psy.unregister_plotter('test_plotter')

    def test_plot_creation_06_list(self):
        """Test the plot creation with a plotter that takes a list of arrays"""
        psy.register_plotter('test_plotter',
                             import_plotter=True, module='test_plotter',
                             plotter_name='TestPlotter', prefer_list=True)
        ds = psy.open_dataset(bt.get_file('test-t2m-u-v.nc'))
        # test the creation of one list
        # psyplot.project.Project([arr4: psyplot.data.InteractiveList([
        #     arr0: 4-dim DataArray of t2m, with
        #         (time, lev, lat, lon)=(5, 4, 96, 192), ,
        #     arr1: 4-dim DataArray of u, with
        #         (time, lev, lat, lon)=(5, 4, 96, 192), ])])
        sp = psy.plot.test_plotter(ds, name=['t2m', 'u'])
        self.assertEqual(len(sp), 1)
        self.assertEqual(len(sp[0]), 2)
        self.assertEqual(sp[0][0].name, 't2m')
        self.assertEqual(sp[0][1].name, 'u')
        self.assertEqual(sp[0][0].shape, ds.t2m.shape)
        self.assertEqual(sp[0][1].shape, ds.u.shape)
        self.assertEqual(sp[0][0].values.tolist(), ds.t2m.values.tolist())
        self.assertEqual(sp[0][1].values.tolist(), ds.u.values.tolist())
        psy.close()
        psy.unregister_plotter('test_plotter')

    def test_plot_creation_07_list_and_dims(self):
        # use dimensions which should result in one list with 4 arrays,
        # t2m, t2m, u, u
        # psyplot.project.Project([arr3: psyplot.data.InteractiveList([
        #     arr0: 3-dim DataArray of t2m, with
        #         (time, lev, lat)=(5, 4, 96), lon=1.875,
        #     arr1: 3-dim DataArray of t2m, with
        #         (time, lev, lat)=(5, 4, 96), lon=3.75,
        #     arr2: 3-dim DataArray of u, with
        #         (time, lev, lat)=(5, 4, 96), lon=1.875,
        #     arr3: 3-dim DataArray of u, with
        #         (time, lev, lat)=(5, 4, 96), lon=3.75])])
        psy.register_plotter('test_plotter',
                             import_plotter=True, module='test_plotter',
                             plotter_name='TestPlotter', prefer_list=True)
        ds = psy.open_dataset(bt.get_file('test-t2m-u-v.nc'))
        sp = psy.plot.test_plotter(ds, name=['t2m', 'u'], x=[1, 2])
        self.assertEqual(len(sp), 1)
        self.assertEqual(len(sp[0]), 4)
        self.assertEqual(sp[0][0].name, 't2m')
        self.assertEqual(sp[0][1].name, 't2m')
        self.assertEqual(sp[0][2].name, 'u')
        self.assertEqual(sp[0][3].name, 'u')
        self.assertEqual(sp[0][0].shape, ds.t2m.isel(lon=1).shape)
        self.assertEqual(sp[0][1].shape, ds.t2m.isel(lon=2).shape)
        self.assertEqual(sp[0][2].shape, ds.u.isel(lon=1).shape)
        self.assertEqual(sp[0][3].shape, ds.u.isel(lon=2).shape)
        self.assertEqual(sp[0][0].values.tolist(),
                         ds.t2m.isel(lon=1).values.tolist())
        self.assertEqual(sp[0][1].values.tolist(),
                         ds.t2m.isel(lon=2).values.tolist())
        self.assertEqual(sp[0][2].values.tolist(),
                         ds.u.isel(lon=1).values.tolist())
        self.assertEqual(sp[0][3].values.tolist(),
                         ds.u.isel(lon=2).values.tolist())
        psy.close()
        psy.unregister_plotter('test_plotter')

    def test_plot_creation_08_list_and_2variables(self):
        # test with arrays out of 2 variables. Should result in a list of
        # two arrays, both should have the two variables 't2m' and 'u'
        # psyplot.project.Project([arr2: psyplot.data.InteractiveList([
        #     arr0: 4-dim DataArray of t2m, u, with
        #         (variable, time, lev, lat)=(2, 5, 4, 96), lon=1.875,
        #     arr1: 4-dim DataArray of t2m, u, with
        #         (variable, time, lev, lat)=(2, 5, 4, 96), lon=3.75])])
        psy.register_plotter('test_plotter',
                             import_plotter=True, module='test_plotter',
                             plotter_name='TestPlotter', prefer_list=True)
        ds = psy.open_dataset(bt.get_file('test-t2m-u-v.nc'))
        sp = psy.plot.test_plotter(ds, name=[[['t2m', 'u']]], x=[1, 2])
        self.assertEqual(len(sp), 1)
        self.assertEqual(len(sp[0]), 2)
        self.assertIn('variable', sp[0][0].dims)
        self.assertIn('variable', sp[0][1].dims)
        self.assertEqual(list(sp[0][0].shape),
                         [2] + list(ds.t2m.isel(lon=1).shape))
        self.assertEqual(list(sp[0][1].shape),
                         [2] + list(ds.u.isel(lon=1).shape))
        self.assertEqual(
            sp[0][0].values.tolist(),
            ds[['t2m', 'u']].to_array().isel(lon=1).values.tolist())
        self.assertEqual(
            sp[0][1].values.tolist(),
            ds[['t2m', 'u']].to_array().isel(lon=2).values.tolist())
        psy.close()
        psy.unregister_plotter('test_plotter')

    def test_plot_creation_09_list_of_list_of_arrays(self):
        # test list of list of arrays
        # psyplot.project.Project([
        #     arr0: psyplot.data.InteractiveList([
        #         arr0: 3-dim DataArray of t2m, with
        #             (time, lev, lat)=(5, 4, 96), lon=1.875,
        #         arr1: 3-dim DataArray of u, with #
        #             (time, lev, lat)=(5, 4, 96), lon=1.875]),
        #     arr1: psyplot.data.InteractiveList([
        #         arr0: 3-dim DataArray of t2m, with
        #             (time, lev, lat)=(5, 4, 96), lon=3.75,
        #         arr1: 3-dim DataArray of u, with
        #             (time, lev, lat)=(5, 4, 96), lon=3.75])])
        psy.register_plotter('test_plotter',
                             import_plotter=True, module='test_plotter',
                             plotter_name='TestPlotter', prefer_list=True)
        ds = psy.open_dataset(bt.get_file('test-t2m-u-v.nc'))
        sp = psy.plot.test_plotter(bt.get_file('test-t2m-u-v.nc'),
                                   name=[['t2m', 'u']], x=[1, 2])
        self.assertEqual(len(sp), 2)
        self.assertEqual(len(sp[0]), 2)
        self.assertEqual(len(sp[1]), 2)
        self.assertEqual(sp[0][0].name, 't2m')
        self.assertEqual(sp[0][1].name, 'u')
        self.assertEqual(sp[1][0].name, 't2m')
        self.assertEqual(sp[1][1].name, 'u')
        self.assertEqual(sp[0][0].shape, ds.t2m.isel(lon=1).shape)
        self.assertEqual(sp[0][1].shape, ds.u.isel(lon=1).shape)
        self.assertEqual(sp[1][0].shape, ds.t2m.isel(lon=2).shape)
        self.assertEqual(sp[1][1].shape, ds.u.isel(lon=2).shape)
        self.assertEqual(sp[0][0].values.tolist(),
                         ds.t2m.isel(lon=1).values.tolist())
        self.assertEqual(sp[0][1].values.tolist(),
                         ds.u.isel(lon=1).values.tolist())
        self.assertEqual(sp[1][0].values.tolist(),
                         ds.t2m.isel(lon=2).values.tolist())
        self.assertEqual(sp[1][1].values.tolist(),
                         ds.u.isel(lon=2).values.tolist())
        psy.close()
        ds.close()
        psy.unregister_plotter('test_plotter')

    def test_plot_creation_10_list_array_and_2variables(self):
        # test list of list with array and an array out of 2 variables
        # psyplot.project.Project([
        #     arr0: psyplot.data.InteractiveList([
        #         arr0: 3-dim DataArray of t2m, with
        #             (time, lev, lat)=(5, 4, 96), lon=1.875,
        #         arr1: 4-dim DataArray of u, v, with
        #             (variable, time, lev, lat)=(2, 5, 4, 96), lon=1.875]),
        #     arr1: psyplot.data.InteractiveList([
        #         arr0: 3-dim DataArray of t2m, with
        #             (time, lev, lat)=(5, 4, 96), lon=1.875,
        #         arr1: 4-dim DataArray of u, v, with
        #             (variable, time, lev, lat)=(2, 5, 4, 96), lon=1.875])])
        psy.register_plotter('test_plotter',
                             import_plotter=True, module='test_plotter',
                             plotter_name='TestPlotter', prefer_list=True)
        ds = psy.open_dataset(bt.get_file('test-t2m-u-v.nc'))
        sp = psy.plot.test_plotter(ds, name=[['t2m', ['u', 'v']]], x=[1, 2])
        self.assertEqual(len(sp), 2)
        self.assertEqual(len(sp[0]), 2)
        self.assertEqual(len(sp[1]), 2)
        self.assertEqual(sp[0][0].name, 't2m')
        self.assertIn('variable', sp[0][1].dims)
        self.assertEqual(sp[0][1].coords['variable'].values.tolist(),
                         ['u', 'v'])
        self.assertEqual(sp[1][0].name, 't2m')
        self.assertIn('variable', sp[1][1].dims)
        self.assertEqual(sp[1][1].coords['variable'].values.tolist(),
                         ['u', 'v'])
        self.assertEqual(sp[0][0].shape, ds.t2m.isel(lon=1).shape)
        self.assertEqual(list(sp[0][1].shape),
                         [2] + list(ds.u.isel(lon=1).shape))
        self.assertEqual(sp[1][0].shape, ds.t2m.isel(lon=2).shape)
        self.assertEqual(list(sp[1][1].shape),
                         [2] + list(ds.u.isel(lon=2).shape))
        self.assertEqual(sp[0][0].values.tolist(),
                         ds.t2m.isel(lon=1).values.tolist())
        self.assertEqual(sp[0][1].values.tolist(),
                         ds[['u', 'v']].isel(lon=1).to_array().values.tolist())
        self.assertEqual(sp[1][0].values.tolist(),
                         ds.t2m.isel(lon=2).values.tolist())
        self.assertEqual(sp[1][1].values.tolist(),
                         ds[['u', 'v']].isel(lon=2).to_array().values.tolist())
        psy.close()
        psy.unregister_plotter('test_plotter')

    def test_check_data(self):
        """Test the :meth:`psyplot.project._PlotterInterface.check_data` method
        """
        from psyplot.plotter import Plotter

        class TestPlotter(Plotter):

            @classmethod
            def check_data(cls, name, dims, is_unstructured):
                checks, messages = super(TestPlotter, cls).check_data(
                    name, dims, is_unstructured)
                self.assertEqual(name, ['t2m'])
                for n, d in zip(name, dims):
                    self.assertEqual(len(d),
                                     len(set(ds.t2m.dims) - {'lev', 'lon'}))
                    self.assertEqual(set(d), set(ds.t2m.dims) - {'lev', 'lon'})

        ds = psy.open_dataset(bt.get_file('test-t2m-u-v.nc'))

        psy.register_plotter('test_plotter', module='nothing',
                             plotter_name='dont_care', plotter_cls=TestPlotter,
                             default_dims={'x': 1}, default_slice=slice(1, 3))

        psy.plot.test_plotter.check_data(ds, 't2m', {'lev': 3})
        psy.unregister_plotter('test_plotter')


@unittest.skipIf(not psy.with_cdo, "Cdo not installed")
class TestCdo(unittest.TestCase):

    def setUp(self):
        psy.close('all')
        plt.close('all')

    def tearDown(self):
        for identifier in list(psy.registered_plotters):
            psy.unregister_plotter(identifier)
        psy.close('all')
        plt.close('all')
        tp.results.clear()

    def test_cdo(self):
        cdo = psy.Cdo()
        sp = cdo.timmean(input=bt.get_file('test-t2m-u-v.nc'),
                         name='t2m', dims=dict(z=[1, 2]))
        with psy.open_dataset(bt.get_file('test-t2m-u-v.nc')) as ds:
            lev = ds.lev.values
        self.assertEqual(len(sp), 2, msg=str(sp))
        self.assertEqual(sp[0].name, 't2m')
        self.assertEqual(sp[1].name, 't2m')
        self.assertEqual(sp[0].lev.values, lev[1])
        self.assertEqual(sp[1].lev.values, lev[2])
        self.assertIsNone(sp[0].psy.plotter)
        self.assertIsNone(sp[1].psy.plotter)

    def test_cdo_plotter(self):
        cdo = psy.Cdo()
        psy.register_plotter('test_plotter', module='test_plotter',
                             plotter_name='TestPlotter')
        sp = cdo.timmean(input=bt.get_file('test-t2m-u-v.nc'),
                         name='t2m', dims=dict(z=[1, 2]),
                         plot_method='test_plotter')
        with psy.open_dataset(bt.get_file('test-t2m-u-v.nc')) as ds:
            lev = ds.lev.values
        self.assertEqual(len(sp), 2, msg=str(sp))
        self.assertEqual(sp[0].name, 't2m')
        self.assertEqual(sp[1].name, 't2m')
        self.assertEqual(sp[0].lev.values, lev[1])
        self.assertEqual(sp[1].lev.values, lev[2])
        self.assertIsInstance(sp[0].psy.plotter, tp.TestPlotter)
        self.assertIsInstance(sp[1].psy.plotter, tp.TestPlotter)


class TestMultipleSubplots(unittest.TestCase):

    def test_one_subplot(self):
        plt.close('all')
        axes = psy.multiple_subplots()
        self.assertEqual(len(axes), 1)
        self.assertEqual(plt.get_fignums(), [1])
        self.assertEqual(len(plt.gcf().axes), 1)
        self.assertIs(axes[0], plt.gcf().axes[0])
        plt.close('all')

    def test_multiple_subplots(self):
        plt.close('all')
        axes = psy.multiple_subplots(2, 2, 3, 5)
        self.assertEqual(len(axes), 5)
        self.assertEqual(plt.get_fignums(), [1, 2])
        self.assertEqual(len(plt.figure(1).axes), 3)
        self.assertEqual(len(plt.figure(2).axes), 2)
        it_ax = iter(axes)
        for ax2 in chain(plt.figure(1).axes, plt.figure(2).axes):
            self.assertIs(next(it_ax), ax2)
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
