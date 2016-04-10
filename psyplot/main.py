import argparse
import pickle
import six
from itertools import chain
from functools import partial
from collections import defaultdict
import yaml
import psyplot
from psyplot.docstring import docstrings
from psyplot.warning import warn
from psyplot.parser import FuncArgParser
import logging


logger = logging.getLogger(__name__)


def main():
    """Main function for usage of psyplot from the command line

    This function creates a parser that parses command lines to the
    :func:`make_plot` functions or (if the ``psyplot_gui`` module is
    present, to the :func:`psyplot_gui.start_app` function)

    Returns
    -------
    psyplot.parser.FuncArgParser
        The parser that has been used from the command line"""
    try:
        from psyplot_gui import get_parser as _get_parser
        parser = _get_parser(create=False)
        parser.create_arguments()
        parser.parse_known_to_func()
    except ImportError:
        logger.debug('Failed to import gui', exc_info=True)
        parser = get_parser(create=False)
        parser.update_arg('output', required=True)
        parser.create_arguments()
        parser.parse_to_func()
    return parser


@docstrings.get_sectionsf('make_plot')
@docstrings.dedent
def make_plot(fnames=[], name=[], dims=None, plot_method=None,
              output=None, project=None, engine=None, formatoptions=None,
              tight=False, rc_file=None):
    """
    Eventually start the QApplication or only make a plot

    Parameters
    ----------
    fnames: list of str
        Either the filenames to show, or, if the `project` parameter is set,
        the a list of `,`-separated filenames to make a mapping from the
        original filename to a new one
    name: list of str
        The variable names to plot if the `output` parameter is set
    dims: dict
        A mapping from coordinate names to integers if the `project` is not
        given
    plot_method: str
        The name of the plot_method to use
    output: str or list of str
        If set, the data is loaded and the figures are saved to the specified
        filename and now graphical user interface is shown
    project: str
        If set, the project located at the given file name is loaded
    engine: str
        The engine to use for opening the dataset (see
        :func:`psyplot.data.open_dataset`)
    formatoptions: dict
        A dictionary of formatoption that is applied to the data visualized by
        the chosen `plot_method`
    tight: bool
        If True/set, it is tried to figure out the tight bbox of the figure and
        adjust the paper size of the `output` to it
    rc_file: str
        The path to a yaml configuration file that can be used to update  the
        :attr:`psyplot.rcParams`
    """

    if project is not None and (name != [] or dims is not None):
        warn('The `name` and `dims` parameter are ignored if the `project`'
             ' parameter is set!')
    if rc_file is not None:
        psyplot.rcParams.load_from_file(rc_file)

    if dims is not None and not isinstance(dims, dict):
        dims = dict(chain(*map(six.iteritems, dims)))

    if len(output) == 1:
        output = output[0]
    if not fnames and not project:
        raise ValueError(
            "Either a filename or a project file must be provided if "
            "the output parameter is set!")
    elif project is None and plot_method is None:
        raise ValueError(
            "A plotting method must be provided if the output parameter "
            "is set and not the project!")
    import psyplot.project as psy
    if project is not None:
        fnames = [s.split(',') for s in fnames]
        single_files = (l[0] for l in fnames if len(l) == 1)
        alternative_paths = defaultdict(lambda: next(single_files, None))
        alternative_paths.update([l for l in fnames if len(l) == 2])
        p = psy.Project.load_project(
            project, alternative_paths=alternative_paths,
            engine=engine)
        if formatoptions is not None:
            p.update(fmt=formatoptions)
        p.export(output, tight=tight)
    else:
        pm = getattr(psy.plot, plot_method, None)
        if pm is None:
            raise ValueError("Unknown plot method %s!" % plot_method)
        p = pm(
            fnames, name=name, dims=dims or {}, engine=engine,
            fmt=formatoptions or {}, mf_mode=True)
        p.export(output, tight=tight)
    return


def get_parser(create=True):
    """Return a parser to make that can be used to make plots or open files
    from the command line

    Returns
    -------
    psyplot.parser.FuncArgParser
        The :class:`argparse.ArgumentParser` instance"""
    #: The parse that is used to parse arguments from the command line
    parser = FuncArgParser(
        description="""
        Load a dataset, make the plot and save the result to a file""",
        epilog=docstrings.get_sections(docstrings.dedents("""
        **Examples**

        Here are some examples on how to use psyplot from the command line.

        Plot the variable ``'t2m'`` in a netCDF file ``'myfile.nc'`` and save
        the plot to ``'plot.pdf'``::

            $ psyplot myfile.nc -n t2m -pm mapplot -o test.pdf

        Create two plots for ``'t2m'`` with the first and second timestep on
        the second vertical level::

            $ psyplot myfile.nc -n t2m  -pm mapplot -o test.pdf -d t,0,1 z,1

        If you have save a project using the
        :meth:`psyplot.project.Project.save_project` method into a file named
        ``'project.pkl'``, you can replot this via::

            $ psyplot -p project.pkl -o test.pdf

        If you use a different dataset than the one you used in the project
        (e.g. ``'other_ds.nc'``), you can replace it via::

            $ psyplot other_dataset.nc -p project.pkl -o test.pdf

        or explicitly via::

            $ psyplot old_ds.nc,other_ds.nc -p project.pkl -o test.pdf

        You can also load formatoptions from a configuration file, e.g.::

            $ echo 'title: my title' > fmt.yaml
            $ psyplot myfile.nc -n t2m  -pm mapplot -fmt fmt.yaml -o test.pdf
        """), 'parser', ['Examples']),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.update_arg('version', short='V', long='version', action='version',
                      version=psyplot.__version__, if_existent=False)

    parser.setup_args(make_plot)

    output_grp = parser.add_argument_group(
        'Output options',
        'Options that only have an effect if the `-o` option is set.')

    parser.update_arg('fnames', positional=True, nargs='*')

    parser.update_arg('name', short='n', nargs='*', metavar='variable_name',
                      const=None)

    parser.update_arg('dims', short='d', nargs='+', type=_load_dims,
                      metavar='dim,val1[,val2[,...]]')

    parser.update_arg('plot_method', short='pm')

    parser.update_arg('output', short='o', group=output_grp)

    parser.update_arg('project', short='p')

    parser.update_arg(
        'formatoptions', short='fmt', type=_load_dict, help="""
        The path to a yaml (``'.yml'`` or ``'.yaml'``) or pickle file
        defining a dictionary of formatoption that is applied to the data
        visualized by the chosen `plot_method`""", metavar='FILENAME')

    parser.update_arg('tight', short='t', group=output_grp)

    parser.update_arg('rc_file', short='rc')
    parser.pop_key('rc_file', 'metavar')

    if create:
        parser.create_arguments()

    return parser


def _load_dict(fname):
    with open(fname) as f:
        if fname.endswith('.yml') or fname.endswith('.yaml'):
            return yaml.load(f)
        return pickle.load(f)


def _load_dims(s):
    s = s.split(',')
    if len(s) > 1:
        return {s[0]: list(map(int, s[1:]))}
    return {}


if __name__ == '__main__':
    main()
