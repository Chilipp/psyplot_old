"""Test module of the :mod:`psyplot.plotter.simple` module"""
import os
from itertools import chain
import unittest
import numpy as np
import matplotlib as mpl
import matplotlib.colors as mcol
import matplotlib.pyplot as plt
from psyplot.plotter.simple import (
    LinePlotter, Simple2DPlotter, BarPlotter, ViolinPlotter)
import psyplot.project as psy
import test_baseplotter as tb
import _base_testing as bt
from psyplot import InteractiveList, ArrayList, open_dataset
from psyplot.compat.mplcompat import bold


class LinePlotterTest(tb.BasePlotterTest):
    """Test class for :class:`psyplot.plotter.simple.LinePlotter`"""

    plot_type = 'line'

    @classmethod
    def setUpClass(cls):
        cls.ds = open_dataset(cls.ncfile)
        cls.data = InteractiveList.from_dataset(
            cls.ds, y=[0, 1], z=0, t=0, name=cls.var, auto_update=True)
        cls.plotter = LinePlotter(cls.data)
        cls.create_dirs()

    def plot(self, **kwargs):
        name = kwargs.pop('name', self.var)
        return psy.plot.lineplot(
            self.ncfile, name=name, t=0, z=0, y=[0, 1], **kwargs)

    def ref_grid(self, close=True):
        """Create reference file for grid formatoption

        Create reference file for
        :attr:`~psyplot.plotter.simple.LinePlotter.grid`
        formatoption"""
        sp = self.plot(grid=True)
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('grid1')))
        sp.update(grid='b')
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('grid2')))
        if close:
            sp.close(True, True)

    def ref_transpose(self, close=True):
        """Create reference file for transpose formatoption

        Create reference file for
        :attr:`~psyplot.plotter.simple.LinePlotter.transpose`
        formatoption"""
        sp = self.plot()
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('transpose1')))
        sp.update(transpose=True)
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('transpose2')))
        if close:
            sp.close(True, True)

    def ref_legend(self, close=True):
        """Create reference file for legend formatoption

        Create reference file for
        :attr:`~psyplot.plotter.simple.LinePlotter.legend`
        formatoption"""
        sp = self.plot(
            legend={'loc': 'upper center', 'bbox_to_anchor': (0.5, -0.05),
                    'ncol': 2})
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('legend')))
        if close:
            sp.close(True, True)

    def ref_xticks(self, close=True):
        """Create reference file for xticks formatoption

        Create reference file for
        :attr:`~psyplot.plotter.simple.LinePlotter.xticks`
        formatoption"""
        sp = psy.plot.lineplot(
            self.ncfile, name=self.var, lon=0, lev=0, lat=[0, 1],
            xticklabels={'major': '%m', 'minor': '%d'},
            xtickprops={'pad': 7.0},
            xticks={'minor': 'week', 'major': 'month'})
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('xticks')))
        if close:
            sp.close(True, True)

    def test_grid(self, *args):
        """Test grid formatoption"""
        args = iter(args)
        self.update(grid=True)
        self.compare_figures(next(args, self.get_ref_file('grid1')))
        self.update(grid='b')
        self.compare_figures(next(args, self.get_ref_file('grid2')))

    def test_xlabel(self):
        """Test xlabel formatoption"""
        self.update(xlabel='{desc}')
        label = self.plotter.ax.xaxis.get_label()
        self.assertEqual(label.get_text(), 'longitude [degrees_east]')
        self.update(labelsize=22, labelweight='bold',
                    labelprops={'ha': 'left'})
        self.assertEqual(label.get_size(), 22)
        self.assertEqual(label.get_weight(), bold)
        self.assertEqual(label.get_ha(), 'left')

    def test_ylabel(self):
        """Test ylabel formatoption"""
        self.update(ylabel='{desc}')
        label = self.plotter.ax.yaxis.get_label()
        self.assertEqual(label.get_text(), 'Temperature [K]')
        self.update(labelsize=22, labelweight='bold',
                    labelprops={'ha': 'left'})
        self.assertEqual(label.get_size(), 22)
        self.assertEqual(label.get_weight(), bold)
        self.assertEqual(label.get_ha(), 'left')

    def test_xlim(self):
        """Test xlim formatoption"""
        curr_lim = self.plotter.ax.get_xlim()
        self.update(xlim=(-1, 300))
        self.assertEqual(self.plotter.ax.get_xlim(), (-1, 300))
        self.update(xlim=(-1, 'rounded'))
        self.assertEqual(self.plotter.ax.get_xlim(), (-1, curr_lim[1]))

    def test_ylim(self):
        """Test ylim formatoption"""
        curr_lim = self.plotter.ax.get_ylim()
        self.update(ylim=(-1, 300))
        self.assertEqual(self.plotter.ax.get_ylim(), (-1, 300))
        self.update(ylim=(-1, 'rounded'))
        self.assertEqual(self.plotter.ax.get_ylim(), (-1, curr_lim[1]))

    def test_color(self):
        current_colors = [l.get_color() for l in self.plotter.ax.lines]
        self.update(color=['y', 'g'])
        self.assertEqual([l.get_color() for l in self.plotter.ax.lines],
                         ['y', 'g'])
        self.update(color=None)
        self.assertEqual([l.get_color() for l in self.plotter.ax.lines],
                         current_colors)

    def test_transpose(self, *args):
        """Test transpose formatoption"""
        args = iter(args)
        self.compare_figures(next(args, self.get_ref_file('transpose1')))
        self.update(transpose=True)
        self.compare_figures(next(args, self.get_ref_file('transpose2')))

    def test_legend(self, *args):
        """Test legend and legendlabels formatoption"""
        args = iter(args)
        self.update(legend=False)
        self.assertIsNone(self.plotter.ax.legend_)
        self.update(legend={
            'loc': 'upper center', 'bbox_to_anchor': (0.5, -0.05), 'ncol': 2})
        self.compare_figures(next(args, self.get_ref_file('legend')))
        self.update(legendlabels='%(lat)s')
        self.assertEqual([t.get_text() for t in plt.gca().legend_.get_texts()],
                         list(map(str, self.data[0].base.lat.values[:2])))

    def test_xticks(self, *args):
        """Test xticks, xticklabels, xtickprops formatoptions"""
        self._test_DataTicksCalculator()
        self._test_DtTicksBase(*args)

    def _test_DataTicksCalculator(self):
        # testing of psyplot.plotter.simple.DataTicksCalculator
        self.update(xticks=['data', 2])
        ax = plt.gca()
        lon = self.data[0].lon.values
        self.assertEqual(list(ax.get_xticks()),
                         list(lon[::2]))
        self.update(xticks=['mid', 2])

        self.assertEqual(
            list(ax.get_xticks()), list(
                (lon[:-1] + lon[1:]) / 2.)[::2])
        self.update(xticks='rounded')
        self.assertEqual(
            list(ax.get_xticks()),
            np.linspace(0, 400, 11, endpoint=True).tolist())
        self.update(xticks='roundedsym')
        self.assertEqual(
            list(ax.get_xticks()),
            np.linspace(-400, 400, 10, endpoint=True).tolist())
        self.update(xticks='minmax')
        self.assertEqual(
            list(ax.get_xticks()), np.linspace(
                self.data[0].lon.values.min(), self.data[0].lon.values.max(),
                11, endpoint=True).tolist())
        self.update(xticks='sym')
        self.assertEqual(
            list(ax.get_xticks()), np.linspace(
                -self.data[0].lon.values.max(), self.data[0].lon.values.max(),
                10, endpoint=True).tolist())

    def _test_DtTicksBase(self, *args):
        # testing of psyplot.plotter.simple.DtTicksBase
        args = iter(args)
        data = InteractiveList.from_dataset(
            self.data[0].base, y=[0, 1], z=0, x=0, name=self.var,
            auto_update=True)
        plotter = self.plotter.__class__(data)
        ax = plotter.ax
        xticks = {'major': ax.get_xticks(), 'minor': ax.get_xticks(minor=True)}
        plotter.update(xticks='month')
        self.assertEqual(list(plt.gca().get_xticks()),
                         [722494.75, 722524.25, 722554.75, 722585.25])
        plotter.update(xticks='monthbegin')
        self.assertEqual(
            list(plt.gca().get_xticks()),
            [722450.75, 722481.75, 722509.75, 722540.75, 722570.75, 722601.75])
        plotter.update(xticks='monthend')
        self.assertEqual(
            list(plt.gca().get_xticks()),
            [722480.75, 722508.75, 722539.75, 722569.75, 722600.75])
        plotter.update(xticks='month', xticklabels='%m')
        # sometimes the labels are only set after drawing
        if ax.get_xticklabels()[0].get_text():
            self.assertEqual(
                [int(t.get_text()) for t in ax.get_xticklabels()[:]],
                list(range(2, 6)))
        plotter.update(xticks={'minor': 'week'}, xticklabels={'minor': '%d'},
                       xtickprops={'pad': 7.0})
        self.assertEqual(
            plotter.ax.get_xticks(minor=True).tolist(),
            [722487.75, 722494.75, 722501.75, 722508.75, 722515.75,
             722522.75, 722529.75, 722536.75, 722543.75, 722550.75,
             722557.75, 722564.75, 722571.75, 722578.75, 722585.75,
             722592.75, 722599.75])
        self.compare_figures(next(args, self.get_ref_file('xticks')))
        plotter.update(xticks={'major': None, 'minor': None})
        self.assertEqual(list(plotter.ax.get_xticks()),
                         list(xticks['major']))
        self.assertEqual(list(plotter.ax.get_xticks(minor=True)),
                         list(xticks['minor']))

    def test_tick_rotation(self):
        """Test xrotation and yrotation formatoption"""
        self.update(xrotation=90, yrotation=90)
        self.assertTrue(all(
            t.get_rotation() == 90 for t in self.plotter.ax.get_xticklabels()))
        self.assertTrue(all(
            t.get_rotation() == 90 for t in self.plotter.ax.get_yticklabels()))

    def test_ticksize(self):
        """Tests ticksize formatoption"""
        self.update(ticksize=24)
        ax = self.plotter.ax
        self.assertTrue(all(t.get_size() == 24 for t in chain(
            ax.get_xticklabels(), ax.get_yticklabels())))
        self.update(
            xticks={'major': ['data', 40], 'minor': ['data', 10]},
            ticksize={'major': 12, 'minor': 10}, xtickprops={'pad': 7.0})
        self.assertTrue(all(t.get_size() == 12 for t in chain(
            ax.get_xticklabels(), ax.get_yticklabels())))
        self.assertTrue(all(
            t.get_size() == 10 for t in ax.get_xticklabels(minor=True)))

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


class ViolinPlotterTest(LinePlotterTest):
    """Test class for :class:`psyplot.plotter.simple.BarPlotter`"""

    plot_type = 'violin'

    @classmethod
    def setUpClass(cls):
        cls.ds = open_dataset(cls.ncfile)
        cls.data = InteractiveList.from_dataset(
            cls.ds, y=[0, 1], z=0, t=0, name=cls.var, auto_update=True)
        cls.plotter = ViolinPlotter(cls.data)
        cls.create_dirs()

    def plot(self, **kwargs):
        name = kwargs.pop('name', self.var)
        return psy.plot.violinplot(
            self.ncfile, name=name, t=0, z=0, y=[0, 1], **kwargs)

    @unittest.skip("No need for figure creation")
    def ref_xticks(self, close=True):
        pass

    @unittest.skip('Test needs to be implemented')
    def test_xticks(self, *args):
        """
        .. todo::

            Implement this test"""
        # TODO: implement this test
        pass

    def test_color(self):
        pass


class BarPlotterTest(LinePlotterTest):
    """Test class for :class:`psyplot.plotter.simple.BarPlotter`"""

    plot_type = 'bar'

    @classmethod
    def setUpClass(cls):
        cls.ds = open_dataset(cls.ncfile)
        cls.data = InteractiveList.from_dataset(
            cls.ds, y=[0, 1], z=0, t=0, name=cls.var, auto_update=True)
        cls.plotter = BarPlotter(cls.data)
        cls.create_dirs()

    def plot(self, **kwargs):
        name = kwargs.pop('name', self.var)
        return psy.plot.barplot(
            self.ncfile, name=name, t=0, z=0, y=[0, 1], **kwargs)

    @unittest.skip("No need for figure creation")
    def ref_xticks(self, close=True):
        pass

    def test_xticks(self, *args):
        self._test_DtTicksBase()

    def _test_DtTicksBase(self, *args):
        data = InteractiveList.from_dataset(
            self.data[0].base, y=[0, 1], z=0, x=0, name=self.var,
            auto_update=True)
        plotter = self.plotter.__class__(data)
        ax = plotter.ax
        plotter.update(xticklabels='%m')
        self.assertListEqual(ax.get_xticks().astype(int).tolist(),
                             list(range(5)))

    def test_color(self):
        current_colors = [
            c[0].get_facecolor() for c in self.plotter.ax.containers]
        self.update(color=['y', 'g'])

        self.assertEqual(
            [c[0].get_facecolor() for c in self.plotter.ax.containers],
            list(map(mcol.colorConverter.to_rgba, ['y', 'g'])))
        self.update(color=None)
        self.assertEqual(
            [c[0].get_facecolor() for c in self.plotter.ax.containers],
            current_colors)


class References2D(object):
    """abstract base class that defines reference methods for 2D plotter"""

    def ref_datagrid(self, close=True):
        """Create reference file for datagrid formatoption

        Create reference file for
        :attr:`~psyplot.plotter.simple.Simple2DPlotter.datagrid`
        formatoption"""
        if self.plot_type[:6] == 'simple':
            kwargs = dict(xlim=(0, 40), ylim=(0, 40))
        else:
            kwargs = dict(lonlatbox='Europe')
        sp = self.plot(**kwargs)
        sp.update(datagrid='k-')
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('datagrid')))
        if close:
            sp.close(True, True)

    def ref_cmap(self, close=True):
        """Create reference file for cmap formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.simple.Simple2DPlotter.cmap`
        formatoption"""
        sp = self.plot(cmap='RdBu')
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('cmap')))
        if close:
            sp.close(True, True)

    def ref_cbar(self, close=True):
        """Create reference file for cbar formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.simple.Simple2DPlotter.cbar`
        formatoption"""
        sp = self.plot(cbar=['fb', 'fr', 'fl', 'ft', 'b', 'r'])
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('cbar')))
        if close:
            sp.close(True, True)

    def ref_miss_color(self, close=True):
        """Create reference file for miss_color formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.simple.Simple2DPlotter.miss_color`
        formatoption"""
        if self.plot_type[:3] == 'map':
            kwargs = {'projection': 'ortho', 'grid_labels': False}
        else:
            kwargs = {}
        sp = self.plot(maskless=280, miss_color='0.9', **kwargs)
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('miss_color')))
        if close:
            sp.close(True, True)

    def ref_cbarspacing(self, close=True):
        """Create reference file for cbarspacing formatoption.

        Create reference file for
        :attr:`~psyplot.plotter.simple.Simple2DPlotter.cbarspacing`
        formatoption"""
        if self.plot_type.endswith('vector') or getattr(self, 'vector_mode',
                                                        False):
            kwargs = dict(
                bounds=np.arange(0, 1.45, 0.1).tolist() + np.linspace(
                    1.5, 13.5, 7, endpoint=True).tolist() + np.arange(
                        13.6, 15.05, 0.1).tolist(), color='absolute')
        else:
            kwargs = dict(bounds=list(range(235, 250)) + np.linspace(
                250, 295, 7, endpoint=True).tolist() + list(range(296, 310)))
        sp = self.plot(
            cbarspacing='proportional', cticks='rounded',
            **kwargs)
        sp.export(os.path.join(bt.ref_dir, self.get_ref_file('cbarspacing')))
        if close:
            sp.close(True, True)


class Simple2DPlotterTest(LinePlotterTest, References2D):

    plot_type = 'simple2D'

    def plot(self, **kwargs):
        name = kwargs.pop('name', self.var)
        return psy.plot.plot2d(self.ncfile, name=name, **kwargs)

    @classmethod
    def setUpClass(cls):
        cls.ds = open_dataset(cls.ncfile)
        cls.data = ArrayList.from_dataset(
            cls.ds, t=0, z=0, name=cls.var, auto_update=True)[0]
        cls.plotter = Simple2DPlotter(cls.data)
        cls.create_dirs()

    @unittest.skip("legend formatoption not implemented for 2D-Plotter")
    def ref_legend(self, *args, **kwargs):
        pass

    @unittest.skip("no need for xticks formatoption reference for 2D-Plotter")
    def ref_xticks(self, *args, **kwargs):
        pass

    @unittest.skip("color formatoption not implemented for 2D-Plotter")
    def test_color(self):
        pass

    def test_ylabel(self):
        """Test ylabel formatoption"""
        self.update(ylabel='{desc}')
        label = self.plotter.ax.yaxis.get_label()
        self.assertEqual(label.get_text(), 'latitude [degrees_north]')
        self.update(labelsize=22, labelweight='bold',
                    labelprops={'ha': 'left'})
        self.assertEqual(label.get_size(), 22)
        self.assertEqual(label.get_weight(), bold)
        self.assertEqual(label.get_ha(), 'left')

    def test_xticks(self):
        """Test xticks formatoption"""
        self._test_DataTicksCalculator()

    def test_extend(self):
        """Test extend formatoption"""
        self.update(extend='both')
        self.assertEqual(self.plotter.cbar.cbars['b'].extend, 'both')
        self.update(extend='min')
        self.assertEqual(self.plotter.cbar.cbars['b'].extend, 'min')
        self.update(extend='neither')
        self.assertEqual(self.plotter.cbar.cbars['b'].extend, 'neither')

    def test_legend(self):
        pass

    def test_cticks(self):
        """Test cticks, cticksize, ctickweight, ctickprops formatoptions"""
        cticks = np.round(
            np.linspace(self.data.values.min(), self.data.values.max(), 11,
                        endpoint=True), decimals=2).tolist()
        self.update(cticks='minmax')
        cbar = self.plotter.cbar.cbars['b']
        self.assertEqual(list(map(
            lambda t: float(t.get_text()), cbar.ax.get_xticklabels())), cticks)
        self.update(cticklabels='%3.1f')
        cticks = np.round(cticks, decimals=1).tolist()
        self.assertEqual(list(map(
            lambda t: float(t.get_text()), cbar.ax.get_xticklabels())), cticks)
        self.update(cticksize=20, ctickweight=bold, ctickprops={
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
        self.update(xlim=(0, 40), ylim=(0, 40), datagrid='k-')
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
        self.update(maskless=280, miss_color='0.9')
        self.compare_figures(next(iter(args), self.get_ref_file('miss_color')))

    def test_cbarspacing(self, *args):
        """Test cbarspacing formatoption"""
        self.update(
            cbarspacing='proportional', cticks='rounded',
            bounds=list(range(235, 250)) + np.linspace(
                250, 295, 7, endpoint=True).tolist() + list(range(296, 310)))
        self.compare_figures(next(iter(args),
                                  self.get_ref_file('cbarspacing')))


class LinePlotterTest2D(tb.TestBase2D, LinePlotterTest):
    """Test :class:`psyplot.plotter.simple.LinePlotter` class without
    time and vertical dimension"""

    var = 't2m_2d'

    def test_xticks(self, *args):
        """Test xticks, xticklabels, xtickprops formatoptions"""
        self._test_DataTicksCalculator()


class Simple2DPlotterTest2D(tb.TestBase2D, Simple2DPlotterTest):
    """Test :class:`psyplot.plotter.simple.Simple2DPlotter` class without
    time and vertical dimension"""

    var = 't2m_2d'

tests2d = [LinePlotterTest2D, Simple2DPlotterTest2D]

# skip the reference creation functions of the 2D Plotter tests
for cls in tests2d:
    skip_msg = "Reference figures for this class are created by the %s" % (
        cls.__name__[:-2])
    for funcname in filter(lambda s: s.startswith('ref'), dir(cls)):
        setattr(cls, funcname, unittest.skip(skip_msg)(lambda self: None))


if __name__ == '__main__':
    bt.RefTestProgram()
