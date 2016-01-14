"""Project module of the psyplot Package

This module contains the :class:`Project` class that serves as the main
part of the psyplot API. One instance of the :class:`Project` class serves as
coordinator of multiple plots and can be distributed into subprojects that
keep reference to the main project without holding all array instances

Furthermore this module contains an easy pyplot-like API to the current
subproject."""
import os
import six
import logging
from importlib import import_module
from itertools import chain, repeat, cycle
from collections import Iterable
from functools import wraps
import xray
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from . import rcParams
from .warning import critical
from .docstring import docstrings, dedent, safe_modulo
from .data import ArrayList, open_dataset, open_mfdataset, sort_kwargs
from .plotter.colors import show_colormaps, get_cmap
from .compat.pycompat import OrderedDict, range

_open_projects = []  # list of open projects
_current_project = None  # current main project
_current_subproject = None  # current subproject


@docstrings.get_sectionsf('multiple_subplots')
def multiple_subplots(rows=1, cols=1, maxplots=None, n=1, delete=True, *args,
                      **kwargs):
    """
    Function to create subplots.

    This function creates so many subplots on so many figures until the
    specified number `n` is reached.

    Parameters
    ----------
    rows: int
        The number of subplots per rows
    cols: int
        The number of subplots per column
    maxplots: int
        The number of subplots per figure (if None, it will be row*cols)
    n: int
        number of subplots to create
    ``*args`` and ``**kwargs``
        anything that is passed to the :func:`matplotlib.pyplot.subplots`
        function
    delete: bool
        If True, the additional subplots per figure are deleted

    Returns
    -------
    list
        list of maplotlib.axes.SubplotBase instances"""
    axes = np.array([])
    maxplots = maxplots or rows * cols
    kwargs.setdefault('figsize', [
        min(8.*cols, 16), min(6.5*rows, 12)])
    for i in range(0, n, maxplots):
        fig, ax = plt.subplots(rows, cols, *args, **kwargs)
        try:
            axes = np.append(axes, ax.ravel()[:maxplots])
            if delete:
                for iax in range(maxplots, rows * cols):
                    fig.delaxes(ax.ravel()[iax])
        except AttributeError:  # got a single subplot
            axes = np.append(axes, [ax])
        if i + maxplots > n and delete:
            for ax2 in axes[n:]:
                fig.delaxes(ax2)
                axes = axes[:n]
    return axes


def _only_main(func):
    """Call the given `func` only from the main project"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.is_main:
            return getattr(self.main, func.__name__)(*args, **kwargs)
        return func(self, *args, **kwargs)
    return wrapper


def _first_main(func):
    """Call the given `func` with the same arguments but after the function
    of the main project"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.is_main:
            getattr(self.main, func.__name__)(*args, **kwargs)
        return func(self, *args, **kwargs)
    return wrapper


class Project(ArrayList):
    """A manager of multiple interactive data projects"""

    _main = None

    @property
    def main(self):
        """:class:`Project`. The main project of this subproject"""
        return self._main if self._main is not None else self

    @main.setter
    def main(self, value):
        self._main = value

    @property
    @dedent
    def plot(self):
        """
        Plotting instance of this :class:`Project`. See the
        :class:`ProjectPlotter` class for method documentations"""
        return self._plot

    @property
    def figs(self):
        """A mapping from figures to data objects with the plotter in this
        figure"""
        ret = OrderedDict()
        for arr in self:
            if arr.plotter:
                fig = arr.plotter.ax.get_figure()
                if fig in ret:
                    ret[fig].append(arr)
                else:
                    ret[fig] = Project([arr], main=self.main)
        return ret

    @property
    def is_main(self):
        """:class:`bool`. True if this :class:`Project` is a main project"""
        return self._main is None

    @property
    def logger(self):
        """:class:`logging.Logger` of this project"""
        if not self.is_main:
            return self.main.logger
        try:
            return self._logger
        except AttributeError:
            self.set_logger()
            return self._logger

    @logger.setter
    def logger(self, value):
        self._logger = value

    @docstrings.get_sectionsf('Project')
    @docstrings.dedent
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        %(ArrayList.parameters)s
        main: Project
            The main project this subproject belongs to (or None if this
            project is the main project)
        num: int
            The number of the project
        """
        self.main = kwargs.pop('main', None)
        self._plot = ProjectPlotter(self)
        self.num = kwargs.pop('num', None)
        super(Project, self).__init__(*args, **kwargs)

    @dedent
    def set_logger(self, name=None, force=False):
        """
        Sets the logging.Logger instance of this plotter.

        Parameters
        ----------
        name: str
            name of the Logger. If None and the :attr:`data` attribute is not
            None, it will be named like <module name>.<arr_name>.<class name>,
            where <arr_name> is the name of the array in the :attr:`data`
            attribute
        force: Bool.
            If False, do not set it if the instance has already a logger
            attribute."""
        if name is None:
            name = '%s.%s.%s' % (self.__module__, self.__class__.__name__,
                                 self.num)
        if not hasattr(self, '_logger') or force:
            self._logger = logging.getLogger(name)
            self.logger.debug('Initializing...')

    @classmethod
    @docstrings.get_sectionsf('Project._register_plotter')
    @dedent
    def _register_plotter(cls, identifier, module, plotter_name,
                          plotter_cls=None):
        """
        Register a plotter in the :class:`Project` class to easy access it

        Parameters
        ----------
        identifier: str
            Name of the attribute that is used to filter for the instances
            belonging to this plotter
        module: str
            The module from where to import the `plotter_name`
        plotter_name: str
            The name of the plotter class in `module`
        plotter_cls: type
            The imported class of `plotter_name`. If None, it will be imported
            when it is needed
        """
        if plotter_cls is not None:  # plotter has already been imported
            def get_x(self):
                return self(plotter_cls)
        else:
            def get_x(self):
                return self(getattr(import_module(module), plotter_name))
        setattr(cls, identifier, property(get_x, doc=(
            "List of data arrays that are plotted by :class:`%s.%s`"
            " plotters") % (module, plotter_name)))

    def disable(self):
        """Disables the plotters in this list"""
        for arr in self:
            if arr.plotter:
                arr.plotter.disabled = True

    def enable(self):
        for arr in self:
            if arr.plotter:
                arr.plotter.disabled = False

    def __call__(self, *args, **kwargs):
        ret = super(Project, self).__call__(*args, **kwargs)
        ret.main = self.main
        return ret

    @_first_main
    def extend(self, *args, **kwargs):
        return super(Project, self).extend(*args, **kwargs)

    extend.__doc__ = ArrayList.extend.__doc__

    @_first_main
    def append(self, *args, **kwargs):
        return super(Project, self).append(*args, **kwargs)

    append.__doc__ = ArrayList.append.__doc__

    __call__.__doc__ = ArrayList.__call__.__doc__

    @docstrings.get_sectionsf('Project.close')
    @dedent
    def close(self, figs=True, data=False):
        """
        Close this project instance

        Parameters
        ----------
        figs: bool
            Close the figures
        data: bool
            delete the arrays from the (main) project"""
        for arr in self[:]:
            if figs and arr.plotter is not None:
                plt.close(arr.plotter.ax.get_figure().number)
            if data:
                self.remove(arr)
                if not self.is_main:
                    self.main.remove(arr)
            arr.plotter = None

    docstrings.keep_params('multiple_subplots.parameters', 'delete')
    docstrings.delete_params('ArrayList.from_dataset.parameters', 'base')
    docstrings.delete_kwargs('ArrayList.from_dataset.other_parameters',
                             kwargs='kwargs')

    @_only_main
    @docstrings.get_sectionsf('Project._add_data')
    @docstrings.dedent
    def _add_data(self, plotter_cls, filename_or_obj, fmt={}, make_plot=True,
                  draw=True, mf_mode=False, ax=None, engine=None, delete=True,
                  *args, **kwargs):
        """
        Extract data from a dataset and visualize it with the given plotter

        Parameters
        ----------
        plotter_cls: type
            The subclass of :class:`psyplot.plotter.Plotter` to use for
            visualization
        filename_or_obj: xray.Dataset or anything for :func:`xray.open_dataset`
            The object (or file name) to open
        fmt: dict
            Formatoptions that shall be when initializing the plot (you can
            however also specify them as extra keyword arguments)
        make_plot: bool
            If True, the data is plotted at the end. Otherwise you have to
            call the :meth:`psyplot.plotter.Plotter.initialize_plot` method by
            yourself
        %(InteractiveBase.start_update.parameters.draw)s
        mf_mode: bool
            If True, the :func:`psyplot.open_mfdataset` method is used.
            Otherwise we use the :func:`psyplot.open_dataset` method which can
            open only one single dataset
        ax: None, tuple (x, y[, z]) or (list of) matplotlib.axes.Axes
            Specifies the subplots on which to plot the new data objects.

            - If None, a new figure will be created for each created plotter
            - If tuple (x, y[, z]), `x` specifies the number of rows, `y` the
              number of columns and the optional third parameter `z` the
              maximal number of subplots per figure.
            - If :class:`matplotlib.axes.Axes` (or list of those, e.g. created
              by the :func:`matplotlib.pyplot.subplots` function), the data
              will be plotted on these subplots
        %(open_dataset.parameters.engine)s
        %(multiple_subplots.parameters.delete)s
        %(ArrayList.from_dataset.parameters.no_base)s

        Other Parameters
        ----------------
        %(ArrayList.from_dataset.other_parameters.no_args_kwargs)s
        ``**kwargs``
            Any other dimension or formatoption that shall be passed to `dims`
            or `fmt` respectively."""
        if not isinstance(filename_or_obj, xray.Dataset):
            if mf_mode:
                filename_or_obj = open_mfdataset(filename_or_obj)
            else:
                filename_or_obj = open_dataset(filename_or_obj)
        fmt = dict(fmt)
        additional_fmt, kwargs = sort_kwargs(
            kwargs, plotter_cls._get_formatoptions())
        fmt.update(additional_fmt)
        # create the subproject
        sub_project = self.from_dataset(
            filename_or_obj, **kwargs)
        self.extend(sub_project, new_name=None, force=True)
        sub_project.main = self
        sub_project.auto_update = sub_project.auto_update or self.auto_update
        scp(sub_project)
        # create the subplots
        if isinstance(ax, tuple):
            axes = iter(multiple_subplots(
                *ax, n=len(sub_project), subplot_kw={
                    'projection': plotter_cls._get_sample_projection()}
                ))
        elif ax is None or isinstance(ax, mpl.axes.SubplotBase):
            axes = repeat(ax)
        else:
            axes = iter(ax)
        for arr in sub_project:
            plotter_cls(arr, make_plot=make_plot, draw=False, ax=next(axes),
                        clear=isinstance(ax, tuple), project=self, **fmt)
        if draw:
            sub_project.draw()
        return sub_project

    def __getitem__(self, key):
        """Overwrites lists __getitem__ by returning subproject if `key` is a
        slice"""
        if isinstance(key, slice):  # return a new project
            return self.__class__(
                super(Project, self).__getitem__(key), main=self.main)
        else:  # return the item
            return super(Project, self).__getitem__(key)

    if six.PY2:  # for compatibility to python 2.7
        def __getslice__(self, *args):
            return self[slice(*args)]

    def show(self):
        """Shows all open figures"""
        plt.show(block=False)

    def joined_attrs(self, delimiter=', ', enhanced=True):
        """Join the attributes of the arrays in this project

        Parameters
        ----------
        delimiter: str
            The string that shall be used as the delimiter in case that there
            are multiple values for one attribute in the arrays
        enhanced: bool
            If True, the :meth:`psyplot.plotter.Plotter.get_enhanced_attrs`
            method is used, otherwise the :attr:`xray.DataArray.attrs`
            attribute is used.

        Returns
        -------
        dict
            A mapping from the attribute to the joined attributes which are
            either strings or (if there is only one attribute value), the
            data type of the corresponding value"""
        if enhanced:
            all_attrs = [arr.plotter.get_enhanced_attrs(
                arr.plotter.data) for arr in self]
        else:
            all_attrs = [arr.attrs for arr in self]
        all_keys = set(chain(*(attrs.keys() for attrs in all_attrs)))
        ret = {}
        for key in all_keys:
            vals = {attrs.get(key, None) for attrs in all_attrs} - {None}
            if len(vals) == 1:
                ret[key] = next(iter(vals))
            else:
                ret[key] = delimiter.join(map(str, vals))
        return ret

    def export(self, output, tight=False, concat=True, close_pdf=None,
               **kwargs):
        """Exports the figures of the project to one or more image files

        Parameters
        ----------
        output: str, iterable or matplotlib.backends.backend_pdf.PdfPages
            if string or list of strings, those define the names of the output
            files. Otherwise you may provide an instance of
            :class:`matplotlib.backends.backend_pdf.PdfPages` to save the
            figures in it.
            If string (or iterable of strings), attribute names in the
            xray.DataArray.attrs attribute as well as index dimensions
            are replaced by the respective value (see examples below).
            Furthermore a single format string without key (e.g. %i, %s, %d,
            etc.) is replaced by a counter.
        tight: bool
            If True, it is tried to figure out the tight bbox of the figure
            (same as bbox_inches='tight')
        concat: bool
            if True and the output format is `pdf`, all figures are
            concatenated into one single pdf
        close_pdf: bool or None
            If True and the figures are concatenated into one single pdf,
            the resulting pdf instance is closed. If False it remains open.
            If None and `output` is a string, it is the same as
            ``close_pdf=True``, if None and `output` is neither a string nor an
            iterable, it is the same as ``close_pdf=False``
        ``**kwargs``
            Any valid keyword for the :func:`matplotlib.pyplot.savefig`
            function

        Returns
        -------
        matplotlib.backends.backend_pdf.PdfPages or None
            a PdfPages instance if output is a string and close_pdf is False,
            otherwise None

        Examples
        --------
        Simply save all figures into one single pdf::

            >>> p = syp.gcp()
            >>> p.export('my_plots.pdf')

        Save all figures into separate pngs with increasing numbers (e.g.
        ``'my_plots_1.png'``)::

            >>> p.export('my_plots_%i.png')

        Save all figures into separate pngs with the name of the variables
        shown in each figure (e.g. ``'my_plots_t2m.png'``)::

            >>> p.export('my_plots_%(name)s.png')

        Save all figures into separate pngs with the name of the variables
        shown in each figure and with increasing numbers (e.g.
        ``'my_plots_1_t2m.png'``)::

            >>> p.export('my_plots_%i_%(name)s.png')

        Specify the names for each figure directly via a list::

            >>> p.export(['my_plots1.pdf', 'my_plots2.pdf'])
        """
        if tight:
            kwargs['bbox_inches'] = 'tight'
        if isinstance(output, six.string_types):  # a single string
            out_fmt = kwargs.pop('format', os.path.splitext(output))[1][1:]
            if out_fmt.lower() == 'pdf' and concat:
                pdf = PdfPages(safe_modulo(output, self.joined_attrs('-')))

                def save(fig):
                    pdf.savefig(fig, **kwargs)

                def close():
                    if close_pdf is None or close_pdf:
                        pdf.close()
                        return
                    return pdf
            else:
                def save(fig):
                    plt.figure(fig.number)
                    try:
                        out = safe_modulo(output, i, print_warning=False)
                    except TypeError:
                        out = output
                    plt.savefig(safe_modulo(
                        out, self.figs[fig].joined_attrs('-')), **kwargs)

                def close():
                    pass
        elif isinstance(output, Iterable):  # a list of strings
            output = cycle(output)

            def save(fig):
                try:
                    out = safe_modulo(next(output), i, print_warning=False)
                except TypeError:
                    out = output
                plt.savefig(safe_modulo(out, self.figs[fig].joined_attrs('-')),
                            **kwargs)

            def close():
                pass
        else:  # an instances of matplotlib.backends.backend_pdf.PdfPages
            def save(fig):
                output.savefig(fig, **kwargs)

            def close():
                if close_pdf:
                    output.close()
        for i, fig in enumerate(self.figs, 1):
            save(fig)
        return close()


class _PlotterInterface(object):
    """Base class for visualizing a data array from an predefined plotter"""

    @property
    def project(self):
        return self._project if self._project is not None else gcp(True)

    @property
    def plotter_cls(self):
        """The plotter class"""
        return self._plotter_cls or getattr(
            import_module(self.module), self.plotter_name)

    _prefer_list = False
    _default_slice = None
    _default_dims = {}

    _print_func = None

    @property
    def print_func(self):
        """The function that is used to return a formatoption

        By default the :func:`print` function is used (i.e. it is printed to
        the terminal)"""
        return self._print_func or six.print_

    @print_func.setter
    def print_func(self, value):
        self._print_func = value

    def __init__(self, methodname, module, plotter_name, project=None):
        self._method = methodname
        self._project = project
        self.module = module
        self.plotter_name = plotter_name

    def __call__(self, *args, **kwargs):
        return self.project._add_data(
            self.plotter_cls, *args, prefer_list=self._prefer_list,
            default_slice=self._default_slice, **dict(chain(
                six.iteritems(self._default_dims), six.iteritems(kwargs))))

    def __getattr__(self, attr):
        if attr in self.plotter_cls._get_formatoptions():
            return self.print_func(getattr(self.plotter_cls, attr).__doc__)
        else:
            raise AttributeError(
                "%s instance does not have a %s attribute" % (
                    self.__class__.__name__, attr))

    def __get__(self, instance, owner):
        """okay"""
        if instance is None:
            return self
        else:
            try:
                return getattr(instance, '_' + self._method)
            except AttributeError:
                setattr(instance, '_' + self._method, self.__class__(
                    self._method, self.module, self.plotter_name,
                    instance._project))
                return getattr(instance, '_' + self._method)

    def __set__(self, instance, value):
        """Actually not required. We just implement it to ensure the python
        "help" function works well"""
        setattr(instance, '_' + self._method, value)

    def __dir__(self):
        return sorted(chain(dir(self.__class__), self.__dict__,
                            self.plotter_cls._get_formatoptions()))

    @docstrings.dedent
    def keys(self, *args, **kwargs):
        """
        Classmethod to return a nice looking table with the given formatoptions

        Parameters
        ----------
        %(Plotter.show_keys.parameters)s

        Returns
        -------
        %(Plotter.show_keys.returns)s

        See Also
        --------
        summaries, docs"""
        return self.plotter_cls.show_keys(*args, **kwargs)

    @docstrings.dedent
    def summaries(self, *args, **kwargs):
        """
        Method to print the summaries of the formatoptions

        Parameters
        ----------
        %(Plotter.show_keys.parameters)s

        Returns
        -------
        %(Plotter.show_keys.returns)s

        See Also
        --------
        keys, docs"""
        return self.plotter_cls.show_summaries(*args, **kwargs)

    @docstrings.dedent
    def docs(self, *args, **kwargs):
        """
        Method to print the full documentations of the formatoptions

        Parameters
        ----------
        %(Plotter.show_keys.parameters)s

        Returns
        -------
        %(Plotter.show_keys.returns)s

        See Also
        --------
        keys, docs"""
        return self.plotter_cls.show_docs(*args, **kwargs)


class ProjectPlotter(object):
    """Plotting methods of the :class:`psyplot.project.Project` class"""

    @property
    def project(self):
        return self._project if self._project is not None else gcp(True)

    def __init__(self, project=None):
        self._project = project

    docstrings.keep_params('ArrayList.from_dataset.parameters',
                           'default_slice')
    docstrings.delete_params('Project._add_data.parameters', 'plotter_cls')

    @classmethod
    @docstrings.get_sectionsf('ProjectPlotter._register_plotter')
    @docstrings.dedent
    def _register_plotter(cls, identifier, module, plotter_name,
                          plotter_cls=None, summary='', prefer_list=False,
                          default_slice=None, default_dims={},
                          show_examples=True,
                          example_call="filename, name=['my_variable'], ..."):
        """
        Register a plotter for making plots

        This class method registeres a plot function for the :class:`Project`
        class under the name of the given `identifier`

        Parameters
        ----------
        %(Project._register_plotter.parameters)s

        Other Parameters
        ----------------
        prefer_list: bool
            Determines the `prefer_list` parameter in the `from_dataset`
            method. If True, the plotter is expected to work with instances of
            :class:`psyplot.InteractiveList` instead of
            :class:`psyplot.InteractiveArray`.
        %(ArrayList.from_dataset.parameters.default_slice)s
        default_dims: dict
            Default dimensions that shall be used for plotting (e.g.
            {'x': slice(None), 'y': slice(None)} for longitude-latitude plots)
        show_examples: bool, optional
            If True, examples how to access the plotter documentation are
            included in class documentation
        example_call: str, optional
            The arguments and keyword arguments that shall be included in the
            example of the generated plot method. This call will then appear as
            ``>>> syp.plot.%%(identifier)s(%%(example_call)s)`` in the
            documentation
        """
        full_name = '%s.%s' % (module, plotter_name)
        if plotter_cls is not None:  # plotter has already been imported
            docstrings.params['%s.formatoptions' % (full_name)] = \
                plotter_cls.show_keys(
                    indent=4, func=str,
                    # include links in sphinx doc
                    _fmt_links=True)
            doc_str = ('    Possible formatoptions are\n\n'
                       '%%(%s.formatoptions)s') % full_name
        else:
            doc_str = ''

        summary = summary or 'Open and plot data via :class:`%s` plotters'

        class PlotMethod(_PlotterInterface):
            __doc__ = docstrings.dedents("""
            %s

            This attribute adds data arrays and plots them via
            :class:`%s` plotters

            To plot data from a netCDF file type::

                >>> syp.plot.%s(%s)

            Parameters
            ----------
            %%(Project._add_data.parameters.no_plotter_cls)s

            Other Parameters
            ----------------
            %%(Project._add_data.other_parameters)s
            %s

            Returns
            -------
            Project
                The subproject that contains the new data array visualized by
                instances of :class:`%s`""" % (
                    summary, full_name, identifier, example_call, doc_str,
                    full_name) + ('' if not show_examples else """

            Notes
            -----
            To explore the formatoptions and their documentations, use the
            ``keys``, ``summaries`` and ``docs`` methods. For example

            .. ipython::

                In [1]: import psyplot.project as syp

                # show the keys corresponding to a group or multiple
                # formatopions
                In [2]: syp.plot.%(id)s.keys('labels')

                # show the summaries of a group of formatoptions or of a
                # formatoption
                In [3]: syp.plot.%(id)s.summaries('title')

                # show the full documentation
                In [4]: syp.plot.%(id)s.docs('plot')

                # or access the documentation via the attribute
                In [5]: print(syp.plot.%(id)s.plot)""" % {'id': identifier})
            )

            _default_slice = default_slice
            _default_dims = default_dims
            _plotter_cls = plotter_cls
            _prefer_list = prefer_list

        setattr(cls, identifier, PlotMethod(identifier, module, plotter_name))


@dedent
def gcp(main=False):
    """
    Get the current project

    Parameters
    ----------
    main: bool
        If True, the current main project is returned, otherwise the current
        subproject is returned.
    See Also
    --------
    scp: Sets the current project
    project: Creates a new project"""
    if main:
        return project() if _current_project is None else _current_project
    else:
        return gcp(True) if _current_subproject is None else \
            _current_subproject


@dedent
def scp(project):
    """
    Set the current project

    Parameters
    ----------
    project: Project
        The project class. If it is a sub project (see
        :attr:`Project.is_main`), the current subproject is set to this
        project. Otherwise it replaces the current main project

    See Also
    --------
    gcp: Returns the current project
    project: Creates a new project"""
    if project is None:
        _scp(project)
    elif not project.is_main:
        _scp(project)
        _scp(project.main, True)
    else:
        _scp(project, True)


def _scp(project, main=False):
    """scp version that allows a bit more control over whether the project is a
    main project or not"""
    global _current_subproject
    global _current_project
    if not main:
        _current_subproject = project
    else:
        _current_project = project


docstrings.delete_params('Project.parameters', 'num')


@docstrings.dedent
def project(num=None, *args, **kwargs):
    """
    Create a new main project

    Parameters
    ----------
    num: int
        The number of the project
    %(Project.parameters.no_num)s

    Returns
    -------
    Project
        The with the given `num` (if it does not already exist, it is created)

    See Also
    --------
    scp: Sets the current project
    gcp: Returns the current project
    """
    numbers = [project.num for project in _open_projects]
    if num in numbers:
        return _open_projects[numbers.index(num)]
    if num is None:
        num = max(numbers) + 1 if numbers else 1
    project = Project(*args, num=num, **kwargs)
    _open_projects.append(project)
    scp(project)
    return project


def close(num=None, *args, **kwargs):
    """
    Close the project

    This method closes the current project or the project specified by `num`

    Parameters
    ----------
    num: int, None or 'all'
        if :class:`int`, it specifies the number of the project, if None, the
        current subproject is closed, if ``'all'``, all open projects are
        closed

    Other Parameters
    ----------------
    %(Project.close.parameters)s"""
    if num is None:
        project = gcp()
        scp(None)
    elif num == 'all':
        for project in _open_projects[:]:
            project.close(*args, **kwargs)
            del _open_projects[0]
    else:
        project = [project for project in _open_projects
                   if project.num == num][0]
        _open_projects.remove(project)
        if _open_projects:
            # set last opened project to the current
            scp(_open_projects[-1])
        else:
            _scp(None, True)  # set the current project to None
    project.close(*args, **kwargs)


docstrings.delete_params('Project._register_plotter.parameters', 'plotter_cls')


@docstrings.dedent
def register_plotter(identifier, module, plotter_name, sorter=True,
                     plot_func=True, import_plotter=None, **kwargs):
    """
    Register a :class:`psyplot.plotter.Plotter` for the projects

    This function registers plotters for the :class:`Project` class to allow
    a dynamical handling of different plotter classes.

    Parameters
    ----------
    %(Project._register_plotter.parameters.no_plotter_cls)s
    sorter: bool, optional
        If True, the :class:`Project` class gets a new property with the name
        of the specified `identifier` which allows you to access the instances
        that are plotted by the specified `plotter_name`
    plot_func: bool, optional
        If True, the :class:`ProjectPlotter` (the class that holds the
        plotting method for the :class:`Project` class and can be accessed via
        the :attr:`Project.plot` attribute) gets an additional method to plot
        via the specified `plotter_name` (see `Other Parameters` below.)
    import_plotter: bool, optional
        If True, the plotter is automatically imported, otherwise it is only
        imported when it is needed. If `import_plotter` is None, then it is
        determined by the :attr:`psyplot.rcParams` ``'project.auto_import'``
        item.

    Other Parameters
    ----------------
    %(ProjectPlotter._register_plotter.other_parameters)s
    """
    if ((import_plotter is None and rcParams['project.auto_import']) or
            import_plotter):
        try:
            plotter_cls = getattr(import_module(module), plotter_name)
        except Exception as e:
            critical("Could not import %s!\n" % module +
                     e.message if six.PY2 else str(e))
            return
    else:
        plotter_cls = None
    if sorter:
        Project._register_plotter(
            identifier, module, plotter_name, plotter_cls)
    if plot_func:
        ProjectPlotter._register_plotter(
            identifier, module, plotter_name, plotter_cls, **kwargs)
    return

for _identifier, _plotter_settings in rcParams['project.plotters'].items():
    register_plotter(_identifier, **_plotter_settings)


plot = ProjectPlotter()
""":class:`ProjectPlotter` of the current project. See the class documentation
for available plotting methods"""
