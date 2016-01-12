import _base_testing as bt
from psyplot.plotter import Plotter, Formatoption, docstrings
from psyplot.data import InteractiveArray


@docstrings.save_docstring('_testing.SimpleFmt')
@docstrings.get_sectionsf('_testing.SimpleFmt')
class SimpleFmt(Formatoption):
    """
    Just a simple formatoption to check the sharing possibility

    Possible types
    --------------
    str
        The string to use in the text"""

    pos = [0.5, 0.5]

    group = 'labels'

    default = ''

    _validate = str

    def update(self, value):
        if not value:
            self.remove()
            return
        self._text = self.ax.text(self.pos[0], self.pos[1], value,
                                  transform=self.ax.transAxes)

    def remove(self):
        if hasattr(self, '_text'):
            self._text.remove()
            del self._text


class SimpleFmt2(SimpleFmt):
    """%(_testing.SimpleFmt)s"""

    pos = [1.0, 1.0]


class SimpleFmt3(SimpleFmt):
    """
    Third test to check the sharing by groups

    Possible types
    --------------
    %(_testing.SimpleFmt.possible_types)s"""

    pos = [0., 0.]

    group = 'colors'


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


class PlotterTest(bt.PsyPlotTestCase):
    """TestCase for testing the Plotter-Formatoption framework"""

    def test_docstring(self):
        """Testing the docstring processing of formatoptions"""
        self.assertEqual(SimpleFmt.__doc__, SimpleFmt2.__doc__)
        self.assertEqual(SimpleFmt3.__doc__, ref_docstring)

    def test_shared(self):
        """Testing the sharing of formatoptions"""
        plotter1 = TestPlotter(InteractiveArray([]))
        plotter2 = TestPlotter(InteractiveArray([]))
        # test sharing of two formatoptions
        plotter1.share(plotter2, ['fmt1', 'fmt3'])
        plotter1.update(fmt1='okay', fmt3='not okay')
        self.assertTrue(hasattr(plotter2.fmt1, '_text'))
        self.assertEqual(plotter2.fmt1._text.get_text(), 'okay')
        self.assertTrue(hasattr(plotter2.fmt3, '_text'))
        self.assertEqual(plotter2.fmt3._text.get_text(), 'not okay')
        # unshare the formatoptions
        plotter1.unshare(plotter2)
        self.assertFalse(hasattr(plotter2.fmt1, '_text'))
        self.assertFalse(hasattr(plotter2.fmt3, '_text'))
        # test sharing of a group of formatoptions
        plotter1.share(plotter2, 'labels')
        plotter1.update(fmt1='okay', fmt2='not okay')
        self.assertTrue(hasattr(plotter2.fmt1, '_text'))
        self.assertEqual(plotter2.fmt1._text.get_text(), 'okay')
        self.assertTrue(hasattr(plotter2.fmt2, '_text'))
        self.assertEqual(plotter2.fmt2._text.get_text(), 'not okay')
        # unshare the plotter
        plotter2.unshare_me('fmt1')
        self.assertTrue(not hasattr(plotter2.fmt1, '_text'))
        self.assertTrue(hasattr(plotter2.fmt2, '_text'))
        self.assertEqual(plotter2.fmt2._text.get_text(), 'not okay')
        plotter2.unshare_me('labels')
        self.assertTrue(not hasattr(plotter2.fmt2, '_text'))


if __name__ == '__main__':
    bt.RefTestProgram()
