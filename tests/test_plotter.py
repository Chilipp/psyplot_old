"""Test module of the basic functionality in the :mod:`psyplot.plotter` module
"""
import unittest
import os.path as osp
from psyplot.compat.pycompat import OrderedDict
import psyplot.data as psyd
import xarray as xr
from psyplot.plotter import Plotter, Formatoption, docstrings, rcParams
from psyplot.config import setup_logging

setup_logging(osp.join(osp.dirname(__file__), 'logging.yml'))


results = OrderedDict()


@docstrings.save_docstring('_testing.SimpleFmt')
@docstrings.get_sectionsf('_testing.SimpleFmt')
class SimpleFmt(Formatoption):
    """
    Just a simple formatoption to check the sharing possibility

    Possible types
    --------------
    str
        The string to use in the text"""

    group = 'labels'

    children = ['fmt2']

    dependencies = ['fmt3']

    @property
    def default(self):
        try:
            return super(SimpleFmt, self).default
        except KeyError:
            return ''

    _validate = str

    def update(self, value):
        key = '%s.%s' % (self.plotter.data.psy.arr_name, self.key)
        if not value:
            results.pop(key, None)
        else:
            results[key] = value


class SimpleFmt2(SimpleFmt):
    """%(_testing.SimpleFmt)s"""

    children = ['fmt3']

    dependencies = []


class SimpleFmt3(SimpleFmt):
    """
    Third test to check the sharing by groups

    Possible types
    --------------
    %(_testing.SimpleFmt.possible_types)s"""

    pos = [0., 0.]

    group = 'something'

    children = dependencies = []


ref_docstring = """Third test to check the sharing by groups

Possible types
--------------
str
    The string to use in the text"""


class TestPlotter(Plotter):
    """A simple Plotter for testing the plotter-formatoption framework"""

    fmt1 = SimpleFmt('fmt1')
    fmt2 = SimpleFmt2('fmt2')
    fmt3 = SimpleFmt3('fmt3')


class PlotterTest(unittest.TestCase):
    """TestCase for testing the Plotter-Formatoption framework"""

    def tearDown(self):
        results.clear()
        rcParams.clear()
        rcParams.update(**{
            key: val[0] for key, val in rcParams.defaultParams.items()})

    def test_docstring(self):
        """Testing the docstring processing of formatoptions"""
        self.assertEqual(SimpleFmt.__doc__, SimpleFmt2.__doc__)
        self.assertEqual(SimpleFmt3.__doc__, ref_docstring)

    def test_shared(self):
        """Testing the sharing of formatoptions"""
        plotter1 = TestPlotter(xr.DataArray([]))
        plotter2 = TestPlotter(xr.DataArray([]))
        plotter1.data.psy.arr_name = 'test1'
        plotter2.data.psy.arr_name = 'test2'

        results.clear()
        # test sharing of two formatoptions
        plotter1.share(plotter2, ['fmt1', 'fmt3'])
        plotter1.update(fmt1='okay', fmt3='okay2')
        # check source
        self.assertIn('test1.fmt1', results)
        self.assertEqual(results['test1.fmt1'], 'okay')
        self.assertIn('test1.fmt3', results)
        self.assertEqual(results['test1.fmt3'], 'okay2')
        # checked shared
        self.assertIn('test2.fmt1', results)
        self.assertEqual(results['test2.fmt1'], 'okay')
        self.assertIn('test2.fmt3', results)
        self.assertEqual(results['test2.fmt3'], 'okay2')

        # unshare the formatoptions
        plotter1.unshare(plotter2)
        # check source
        self.assertIn('test1.fmt1', results)
        self.assertEqual(results['test1.fmt1'], 'okay')
        self.assertIn('test1.fmt3', results)
        self.assertEqual(results['test1.fmt3'], 'okay2')
        # check (formerly) shared
        self.assertNotIn('test2.fmt1', results,
                         msg='Value of fmt1: %s, in results: %s' % (
                            plotter2.fmt1.value, results.get('test2.fmt1')))
        self.assertNotIn('test2.fmt3', results,
                         msg='Value of fmt3: %s, in results: %s' % (
                            plotter2.fmt3.value, results.get('test2.fmt3')))

        # test sharing of a group of formatoptions
        plotter1.share(plotter2, 'labels')
        plotter1.update(fmt1='okay', fmt2='okay2')
        # check source
        self.assertIn('test1.fmt1', results)
        self.assertEqual(results['test1.fmt1'], 'okay')
        self.assertIn('test1.fmt2', results)
        self.assertEqual(results['test1.fmt2'], 'okay2')
        # check shared
        self.assertIn('test2.fmt1', results)
        self.assertEqual(results['test2.fmt1'], 'okay')
        self.assertIn('test2.fmt2', results)
        self.assertEqual(results['test2.fmt2'], 'okay2')
        self.assertNotIn('test2.fmt3', results)

        # unshare the plotter
        plotter2.unshare_me('fmt1')
        self.assertNotIn('test2.fmt1', results)
        self.assertIn('test2.fmt2', results)
        plotter2.unshare_me('labels')
        self.assertNotIn('test2.fmt2', results)

    def test_rc(self):
        """Test the default values and validation
        """
        def validate(s):
            return s + 'okay'
        rcParams.defaultParams['plotter.test1.fmt1'] = ('test1', validate)
        rcParams.defaultParams['plotter.test1.fmt2'] = ('test2', validate)
        rcParams.defaultParams['plotter.test1.fmt3'] = ('test3', validate)
        rcParams.defaultParams['plotter.test2.fmt3'] = ('test3.2', validate)
        rcParams.update(**{
            key: val[0] for key, val in rcParams.defaultParams.items()})

        class ThisTestPlotter(TestPlotter):
            _rcparams_string = ['plotter.test1.']

        class ThisTestPlotter2(ThisTestPlotter):
            _rcparams_string = ['plotter.test2.']

        plotter1 = ThisTestPlotter(xr.DataArray([]))
        plotter2 = ThisTestPlotter2(xr.DataArray([]))

        # plotter1
        self.assertEqual(plotter1.fmt1.value, 'test1okay')
        self.assertEqual(plotter1.fmt2.value, 'test2okay')
        self.assertEqual(plotter1.fmt3.value, 'test3okay')
        # plotter2
        self.assertEqual(plotter2.fmt1.value, 'test1okay')
        self.assertEqual(plotter2.fmt2.value, 'test2okay')
        self.assertEqual(plotter2.fmt3.value, 'test3.2okay')

    def test_fmt_connections(self):
        """Test the order of the updates"""
        plotter = TestPlotter(xr.DataArray([]),
                              fmt1='test', fmt2='test2', fmt3='test3')
        plotter.data.psy.arr_name = 'data'

        # check the initialization order
        self.assertEqual(list(results.keys()),
                         ['data.fmt3', 'data.fmt2', 'data.fmt1'])

        # check the connection properties
        self.assertIs(plotter.fmt1.fmt2, plotter.fmt2)
        self.assertIs(plotter.fmt1.fmt3, plotter.fmt3)
        self.assertIs(plotter.fmt2.fmt3, plotter.fmt3)

        # check the update
        results.clear()
        plotter.update(fmt2='something', fmt3='else')
        self.assertEqual(list(results.keys()),
                         ['data.fmt3', 'data.fmt2', 'data.fmt1'])
        self.assertEqual(plotter.fmt1.value, 'test')
        self.assertEqual(plotter.fmt2.value, 'something')
        self.assertEqual(plotter.fmt3.value, 'else')

        self.assertEqual(list(plotter._sorted_by_priority(
                             [plotter.fmt1, plotter.fmt2, plotter.fmt3])),
                         [plotter.fmt3, plotter.fmt2, plotter.fmt1])

    def test_data_props_array(self):
        """Test the data properties of Formatoptions with a DataArray"""
        data = xr.DataArray([])
        plot_data = data.copy(True)
        plotter = TestPlotter(data)
        plotter.plot_data = plot_data

        self.assertIs(plotter.fmt1.raw_data, data)
        self.assertIs(plotter.fmt1.data, plot_data)

    def test_data_props_list(self):
        """Test the data properties of Formatoptions with an InteractiveList"""
        data = psyd.InteractiveList([xr.DataArray([]), xr.DataArray([])])
        plot_data = data.copy(True)
        plot_data.extend([xr.DataArray([]), xr.DataArray([])],
                         new_name=True)
        plotter = TestPlotter(data)
        plotter.plot_data = plot_data
        plot_data = plotter.plot_data  # the data might have been copied

        self.assertIs(plotter.fmt1.raw_data, data)
        self.assertIs(plotter.fmt1.data, plot_data)

        # test with index in list
        plotter.fmt1.index_in_list = 1
        self.assertIs(plotter.fmt1.raw_data, data[1])
        self.assertIs(plotter.fmt1.data, plot_data[1])

        # test with index in list of plot_data outside raw_data
        plotter.fmt1.index_in_list = 3
        self.assertIs(plotter.fmt1.data, plot_data[3])

    def test_decoder(self):
        """Test the decoder property of Formatoptions with a DataArray"""
        data = xr.DataArray([])
        data.psy.init_accessor(decoder=psyd.CFDecoder(data.psy.base))
        plot_data = data.copy(True)
        plotter = TestPlotter(data)
        plotter.plot_data = plot_data

        self.assertIsInstance(plotter.fmt1.decoder, psyd.CFDecoder)
        self.assertIs(plotter.fmt1.decoder, data.psy.decoder)

        # test with index in list of plot_data outside raw_data
        plotter.plot_data_decoder = decoder = psyd.CFDecoder(data.psy.base)
        self.assertIsInstance(plotter.fmt1.decoder, psyd.CFDecoder)
        self.assertIs(plotter.fmt1.decoder, decoder)

    def test_decoder_list(self):
        """Test the decoder property with an InteractiveList"""
        data = psyd.InteractiveList([xr.DataArray([]), xr.DataArray([])])
        plot_data = data.copy(True)
        plot_data.extend([xr.DataArray([]), xr.DataArray([])],
                         new_name=True)
        for arr in data:
            arr.psy.init_accessor(decoder=psyd.CFDecoder(arr.psy.base))
        plotter = TestPlotter(data)
        plotter.plot_data = plot_data
        plot_data = plotter.plot_data  # the data might have been copied

        self.assertIsInstance(plotter.fmt1.decoder, psyd.CFDecoder)
        self.assertIs(plotter.fmt1.decoder, data[0].psy.decoder)

        # test with index in list
        plotter.fmt1.index_in_list = 1
        self.assertIsInstance(plotter.fmt1.decoder, psyd.CFDecoder)
        self.assertIs(plotter.fmt1.decoder, data[1].psy.decoder)

        # test without index in list
        decoder = psyd.CFDecoder(data[0].psy.base)
        plotter.fmt2.decoder = decoder
        for i, d2 in enumerate(plotter.plot_data_decoder):
            self.assertIs(d2, decoder,
                          msg='Decoder %i has been set wrong!' % i)

        # test with index in list of plot_data outside raw_data
        plotter.fmt1.index_in_list = 3
        decoder2 = psyd.CFDecoder(data[0].psy.base)
        plotter.fmt1.decoder = decoder2
        for i, d2 in enumerate(plotter.plot_data_decoder):
            self.assertIs(d2, decoder if i != 3 else decoder2,
                          msg='Decoder %i has been set wrong!' % i)
        self.assertIsInstance(plotter.fmt1.decoder, psyd.CFDecoder)
        self.assertIs(plotter.fmt1.decoder, plotter.plot_data_decoder[3])


if __name__ == '__main__':
    unittest.main()
