import os
from itertools import chain
import _base_testing as bt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from xray import open_dataset
import psyplot
from psyplot.plotter.baseplotter import BasePlotter
from psyplot import InteractiveList
from psyplot.compat.mplcompat import bold


class BasePlotterTest(bt.PsyPlotTestCase):

    @classmethod
    def setUpClass(cls):
        from matplotlib.backends.backend_pdf import PdfPages
        cls.ds = open_dataset('test-t2m-u-v.nc')
        cls.data = InteractiveList.from_dataset(
            cls.ds, lat=[0, 1], lev=0, time=0, name='t2m', auto_update=True)
        cls.plotter = BasePlotter(cls.data)
        if not os.path.isdir(bt.odir):
            os.makedirs(bt.odir)

        cls.pdf = PdfPages('test_%s.pdf' % cls.__name__)

    @classmethod
    def tearDownClass(cls):
        from psyplot.config.rcsetup import defaultParams
        psyplot.rcParams.update(
            **{key: val[0] for key, val in defaultParams.items()})
        cls.ds.close()
#        plt.close(cls.plotter.ax.get_figure().number)
        cls.pdf.close()

    @classmethod
    def tearDown(cls):
        cls.pdf.savefig(plt.gcf())
        cls.data.update(time=0, todefault=True, replot=True)

    def update(self, *args, **kwargs):
        """Update the plotter of this instance"""
        self.plotter.update(*args, **kwargs)

    def _label_test(self, key, label_func):
        kwargs = {
            key: "Test plot at %Y-%m-%d, {tinfo} o'clock of %(long_name)s"}
        self.update(**kwargs)
        self.assertEqual(
            u"Test plot at 1979-01-31, 18:00 o'clock of %s" % (
                self.data.attrs.get('long_name', 'Temperature')),
            label_func().get_text())
        self.data.update(time=1)
        self.assertEqual(
            u"Test plot at 1979-02-28, 18:00 o'clock of %s" % (
                self.data.attrs.get('long_name', 'Temperature')),
            label_func().get_text())
        self.data.update(time=0)

    def test_title(self):
        """Test title, titlesize, titleweight, titleprops formatoptions"""
        def get_title():
            return self.plotter.ax.title
        self._label_test('title', get_title)
        self.update(titlesize=22, titleweight='bold',
                    titleprops={'ha': 'left'})
        self.assertEqual(get_title().get_size(), 22)
        self.assertEqual(get_title().get_weight(), bold)
        self.assertEqual(get_title().get_ha(), 'left')

    def test_figtitle(self):
        """Test figtitle, figtitlesize, figtitleweight, figtitleprops
        formatoptions"""
        def get_figtitle():
            fig = plt.gcf()
            for text in fig.texts:
                if text.get_position() == (0.5, 0.98):
                    return text
        self._label_test('figtitle', get_figtitle)
        self.update(figtitlesize=22, figtitleweight='bold',
                    figtitleprops={'ha': 'left'})
        self.assertEqual(get_figtitle().get_size(), 22)
        self.assertEqual(get_figtitle().get_weight(), bold)
        self.assertEqual(get_figtitle().get_ha(), 'left')

    def test_text(self):
        """Test text formatoption"""
        def get_default_text():
            for text in chain(*self.plotter.text._texts.values()):
                if text.get_position() == tuple(psyplot.rcParams[
                        'texts.default_position']):
                    return text
        self._label_test('text', get_default_text)
        self.update(
            text=(0.5, 0.5, '%(name)s', 'fig', {'fontsize': 16}))
        for t in self.plotter.text._texts['fig']:
            if t.get_position() == (0.5, 0.5):
                text = t
                break
            else:
                text = False
        self.assertTrue(text is not False)
        if not text:
            return
        self.assertEqual(text.get_text(), getattr(self.data, 'name', 't2m'))
        self.assertEqual(text.get_fontsize(), 16)

    def test_maskgreater(self):
        """Test maskgreater formatoption"""
        self.update(maskgreater=250)
        for arr in self.plotter.maskgreater.iter_plotdata:
            self.assertLessEqual(arr.max().values, 250)

    def test_maskgeq(self):
        """Test maskgeq formatoption"""
        self.update(maskgeq=250)
        for arr in self.plotter.maskgeq.iter_plotdata:
            self.assertLessEqual(arr.max().values, 250)

    def test_maskless(self):
        """Test maskless formatoption"""
        self.update(maskless=250)
        for arr in self.plotter.maskless.iter_plotdata:
            self.assertGreaterEqual(arr.min().values, 250)

    def test_maskleq(self):
        """Test maskleq formatoption"""
        self.update(maskleq=250)
        for arr in self.plotter.maskleq.iter_plotdata:
            self.assertGreaterEqual(arr.min().values, 250)

    def test_maskbetween(self):
        """Test maskbetween formatoption"""
        self.update(maskbetween=[250, 251])
        for arr in self.plotter.maskbetween.iter_plotdata:
            data = arr.values[~np.isnan(arr.values)]
            self.assertLessEqual(data[data < 251].max(), 250)
            self.assertGreaterEqual(data[data > 250].max(), 251)

    def test_axiscolor(self):
        """Test axiscolor formatoption"""
        ax = self.plotter.ax
        positions = ['top', 'right', 'left', 'bottom']
        # test updating all to red
        self.update(axiscolor='red')
        self.assertEqual(['red']*4, list(self.plotter['axiscolor'].values()),
                         "Edgecolors are not red but " + ', '.join(
                         self.plotter['axiscolor'].values()))
        # test updating all to the default setup
        self.update(axiscolor=None)
        for pos in positions:
            error = "Edgecolor ({0}) is not the default color ({1})!".format(
                ax.spines[pos].get_edgecolor(), mpl.rcParams['axes.edgecolor'])
            self.assertEqual(mpl.colors.colorConverter.to_rgba(
                                 mpl.rcParams['axes.edgecolor']),
                             ax.spines[pos].get_edgecolor(), msg=error)
            error = "Linewidth ({0}) is not the default width ({1})!".format(
                ax.spines[pos].get_linewidth(), mpl.rcParams['axes.linewidth'])
            self.assertEqual(mpl.rcParams['axes.linewidth'],
                             ax.spines[pos].get_linewidth(), msg=error)
        # test updating only one spine
        self.update(axiscolor={'top': 'red'})
        self.assertEqual((1., 0., 0., 1.0), ax.spines['top'].get_edgecolor(),
                         msg="Axiscolor ({0}) has not been updated".format(
                             ax.spines['top'].get_edgecolor()))
        self.assertGreater(ax.spines['top'].get_linewidth(), 0.0,
                           "Line width of axis is 0!")
        for pos in positions[1:]:
            error = "Edgecolor ({0}) is not the default color ({1})!".format(
                ax.spines[pos].get_edgecolor(), mpl.rcParams['axes.edgecolor'])
            self.assertEqual(mpl.colors.colorConverter.to_rgba(
                                 mpl.rcParams['axes.edgecolor']),
                             ax.spines[pos].get_edgecolor(), msg=error)


if __name__ == '__main__':
    bt.RefTestProgram()
