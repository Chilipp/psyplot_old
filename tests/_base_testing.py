import os
import re
import six
from unittest import TestCase, main


ref_dir = os.path.join(os.path.dirname(__file__), 'reference_figures')

odir = 'psyplot_testresults'


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

    ncfile = 'test-t2m-u-v.nc'

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
        if self.grid_type:
            f = '_'.join(
                ['test', self.plot_type, identifier, self.grid_type]) + '.png'
        else:
            f = '_'.join(['test', self.plot_type, identifier]) + '.png'
        return f

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
