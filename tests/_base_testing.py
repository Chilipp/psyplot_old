import os
import sys
import re
import six
from unittest import TestCase, main
import matplotlib as mpl
import numpy as np

ref_dir = os.path.join(os.path.dirname(__file__), 'reference_figures',
                       'py' + '.'.join(map(str, sys.version_info[:2])),
                       'mpl' + mpl.__version__.rsplit('.', 1)[0])

odir = 'psyplot_testresults'

test_dir = os.path.dirname(__file__)


def _modifed_usage(s):
    l = s.split('\n')
    # get the index where the options definition begins
    i, t = next(((i, t) for i, t in enumerate(l) if re.search('\s*-\w+', t)))
    # get whites spaces in the beginning of the text
    start = re.match('\s+', t)
    start = start.group() if start else ''
    # get white spaces before the description
    startdesc = t.find(re.search('(?<=\s)\w+', t).group())
    # insert reference option description
    new_s = start + '-r, --ref'
    new_s += ' ' * (startdesc - len(new_s)) + (
        "Create reference figures in the `reference_figures` directory")
    l.insert(i+1, new_s)
    # join everything and return
    return '\n'.join(l)


class RefTestProgram(main):
    """Subclass of the usual :attr:`unittest.main` to include reference figures

    This class allows an additional '-r' option to create reference figures"""

    if six.PY2:
        def parseArgs(self, argv):
            if '-r' in argv or '--ref' in argv:
                self.testLoader.testMethodPrefix = 'ref'
            return super(RefTestProgram, self).parseArgs(
                tuple(arg for arg in argv if arg not in ['-r', '--ref']))

        USAGE = _modifed_usage(main.USAGE)

    else:
        _create_references = None

        def _getParentArgParser(self, *args, **kwargs):
            parser = super(RefTestProgram, self)._getParentArgParser(
                *args, **kwargs)
            parser.add_argument(
                '-r', '--ref', action='store_true', dest='_create_references',
                help=('Create reference figures in the `reference_figures` '
                      'directory'))
            self._create_references = False
            return parser

        def createTests(self, *args, **kwargs):
            if self._create_references:
                self.testLoader.testMethodPrefix = 'ref'
            super(RefTestProgram, self).createTests(*args, **kwargs)


class PsyPlotTestCase(TestCase):
    """Base class for testing the psyplot package. It only provides some
    useful methods to compare figures"""

    longMessage = True

    plot_type = None

    grid_type = None

    ncfile = os.path.join(test_dir, 'test-t2m-u-v.nc')

    @classmethod
    def create_dirs(cls):
        if not os.path.exists(ref_dir):
            os.makedirs(ref_dir)
        if not os.path.exists(odir):
            os.makedirs(odir)

    def get_ref_file(self, identifier):
        """
        Gives the name of the reference file for a test

        This staticmethod gives combines the given `plot_type`, `identifier`
        and `grid_type` to form the name of a reference figure

        Parameters
        ----------
        identifier: str
            The unique identifier for the plot (usually the formatoption name)

        Returns
        -------
        str
            The basename of the reference file"""
        identifiers = ['test']
        if self.plot_type is not None:
            identifiers.append(self.plot_type)
        identifiers.append(identifier)
        if self.grid_type is not None:
            identifiers.append(self.grid_type)
        return "_".join(identifiers) + '.png'

    def compare_figures(self, fname, tol=1, **kwargs):
        """Saves and compares the figure to the reference figure with the same
        name"""
        import matplotlib.pyplot as plt
        from matplotlib.testing.compare import compare_images
        plt.savefig(os.path.join(odir, fname), **kwargs)
        results = compare_images(
            os.path.join(ref_dir, fname), os.path.join(odir, fname),
            tol=tol)
        self.assertIsNone(results, msg=results)

    def assertAlmostArrayEqual(self, actual, desired, rtol=1e-07, atol=0,
                               msg=None, **kwargs):
        """Asserts that the two given arrays are almost the same

        This method uses the :func:`numpy.testing.assert_allclose` function
        to compare the two given arrays.

        Parameters
        ----------
        actual : array_like
            Array obtained.
        desired : array_like
            Array desired.
        rtol : float, optional
            Relative tolerance.
        atol : float, optional
            Absolute tolerance.
        equal_nan : bool, optional.
            If True, NaNs will compare equal.
        err_msg : str, optional
            The error message to be printed in case of failure.
        verbose : bool, optional
            If True, the conflicting values are appended to the error message.
        """
        try:
            np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol,
                                       err_msg=msg or '', **kwargs)
        except AssertionError as e:
            self.fail(e.message)
