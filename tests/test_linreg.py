# -*- coding: utf-8 -*-
import _base_testing as bt
from functools import wraps
import unittest
import numpy as np
import xarray
import seaborn as sns  # to set the axes style
import matplotlib.pyplot as plt
import psyplot.data as psyd
from psyplot.plotter.linreg import LinRegPlotter, DensityRegPlotter
from test_simple import DensityPlotterTest


class LinRegPlotterTest(unittest.TestCase):

    default_slope = 3
    default_intercept = 2
    default_n = 500

    plotter_cls = LinRegPlotter

    def tearDown(self):
        plt.close('all')

    @property
    def plot_data(self):
        return self.plotter.plot_data[0]

    @classmethod
    def define_data(cls, slope=None, intercept=None, scatter=0.1,
                    n=None, **kwargs):
        """Set up the data

        Parameters
        ----------
        slope: float
            The slope of the data. If None, defaults to the
            :attr:`default_slope` attribute
        intercept: float
            The y-value for x==0. If None, defaults to the
            :attr:`default_intercept` attribute
        scatter: float
            The range for the random noise. Random noise will be like
            ``y * rand * scatter`` where rand is a normally random number
            between [-1, 1]
        n: int
            The number of data points. If None, defaults to the
            :attr:`default_n` attribute

        Returns
        -------
        psyplot.data.InteractiveArray
            The array with the x- and y-data that can serve as an input for
            the :class:`psyplot.plotter.linreg.LinRegPlotter`
        """
        if slope is None:
            slope = cls.default_slope
        if intercept is None:
            intercept = cls.default_intercept
        if n is None:
            n = cls.default_n
        x = np.linspace(0, 10, n)
        y = intercept + slope * x
        y += y * np.random.randn(n) * scatter
        da = psyd.InteractiveArray(y, name='y', dims=('x', ), coords={
            'x': xarray.Coordinate('x', x)})
        da.base['v'] = da.x.variable
        return psyd.InteractiveList([da])

    @classmethod
    def define_poly_data(cls, a=None, scatter=0.1, n=None, **kwargs):
        """
        Set up the data

        This method uses the function

        .. math::

            y = a^2 * x * (1 - x)

        to generate data with a polynom of order 2.

        Parameters
        ----------
        a: float
            The parameter to use in the above equation
        scatter: float
            The range for the random noise. Random noise will be like
            ``y * rand * scatter`` where rand is a normally random number
            between [-1, 1]
        n: int
            The number of data points. If None, defaults to the
            :attr:`default_n` attribute

        Returns
        -------
        psyplot.data.InteractiveArray
            The array with the x- and y-data that can serve as an input for
            the :class:`psyplot.plotter.linreg.LinRegPlotter`
        """
        def func(x, a):
            return a * a * x * (1 - x)
        if a is None:
            a = 1.0434
        if n is None:
            n = cls.default_n
        x = np.linspace(0, 1, n)
        y = func(x, a)
        y += y * np.random.randn(n) * scatter
        da = psyd.InteractiveList([
            psyd.InteractiveArray(y, name='y', dims=('x', ), coords={
                'x': xarray.Coordinate('x', x)})])
        return da, func

    def test_nonfixed_fit(self):
        '''Test whether the fit works'''
        da = self.define_data()
        self.plotter = self.plotter_cls(da)
        data = self.plot_data
        self.assertGreater(data.rsquared, 0.8)

    def test_fix0(self):
        '''Test with a fix point of 0'''
        da = self.define_data(intercept=0)
        self.plotter = self.plotter_cls(da, fix=0)
        data = self.plot_data
        self.assertEqual(data.intercept, 0)
        self.assertGreater(data.rsquared, 0.8)

    def test_fix1(self):
        '''Test with a fix point at (0, 1)'''
        da = self.define_data(intercept=1)
        self.plotter = self.plotter_cls(da, fix=1)
        data = self.plot_data
        self.assertEqual(data.intercept, 1)
        self.assertGreater(data.rsquared, 0.8)

    def test_legend(self):
        self.test_nonfixed_fit()
        self.plotter.update(
            legendlabels='%(intercept)1.1f + %(slope)1.1f * x, '
                         'R^2=%(rsquared)1.1f')
        t = plt.gca().legend_.get_texts()[0].get_text()
        data = self.plot_data
        d = {
            'intercept': round(data.intercept, 1),
            'slope': round(data.slope, 1),
            'rsquared': round(data.rsquared, 1)}
        self.assertEqual(
            t, '%(intercept)s + %(slope)s * x, R^2=%(rsquared)s' % d)

    def test_ci(self):
        """Test whether the confidence interval is drawn"""
        self.test_nonfixed_fit()
        ax = self.plotter.ax
        err_fmt = self.plotter.error
        self.assertEqual(self.plot_data.shape[0], 3)
        self.assertTrue(hasattr(err_fmt, '_plot') and len(err_fmt._plot) >= 1)
        self.assertTrue(
            all(a in ax.collections for a in err_fmt._plot))

    def test_poly(self):
        """Testing the fit of a polynom"""
        da, func = self.define_poly_data()
        self.plotter = plotter = self.plotter_cls(da, fit=func)
        err = np.sqrt(plotter.fit.fits[0][0, 0])
        self.assertLess(err, 0.01)


class SingleLinRegPlotterTest(LinRegPlotterTest):
    """Test the :class:`psyplot.plotter.linreg.LinRegPlotter` with a single
    data array not in a list"""

    @classmethod
    def define_data(cls, *args, **kwargs):
        return super(SingleLinRegPlotterTest, cls).define_data(
            *args, **kwargs)[0]

    @classmethod
    def define_poly_data(cls, *args, **kwargs):
        da, func =  super(SingleLinRegPlotterTest, cls).define_poly_data(
            *args, **kwargs)
        return da[0], func

    @property
    def plot_data(self):
        return self.plotter.plot_data


class DensityRegPlotterTest(DensityPlotterTest):
    '''Test whether the plot works in combination with the
    :class:`psyplot.plotter.linreg.LinearRegressionPlotter`'''

    @classmethod
    def setUpClass(cls):
        cls.data = cls.define_data()
        cls.plotter = DensityRegPlotter(cls.data)

    @property
    def plot_data(self):
        return self.plotter.plot_data[0]

    @classmethod
    def define_data(cls, *args, **kwargs):
        return SingleLinRegPlotterTest.define_data(*args, **kwargs)


class DensityRegPlotterTestFits(SingleLinRegPlotterTest):
    """Test the fit part of the
    :class:`psyplot.plotter.linreg.DensityRegPlotter` class"""

    plotter_cls = DensityRegPlotter

    @property
    def plot_data(self):
        return self.plotter.plot_data[1]


if __name__ == '__main__':
    unittest.main()
