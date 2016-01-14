"""Default management of the psyplot package

This module defines the necessary classes, data and functions for the default
configuration of the module.
The structure is motivated and to larger parts taken from the matplotlib_
package.

.. _matplotlib: http://matplotlib.org/api/"""
import os
import sys
import six
import re
import yaml
from itertools import repeat
import matplotlib as mpl
from ..warning import warn
from ..compat.pycompat import UserDict, DictMethods, getcwd, zip, isstring, map
from ..compat.mplcompat import mpl_version
from matplotlib.patches import ArrowStyle
from numpy import asarray
from matplotlib.rcsetup import (
    validate_bool, validate_color, validate_bool_maybe_none, validate_fontsize,
    validate_nseq_float, ValidateInStrings, validate_int, validate_colorlist,
    validate_path_exists, validate_legend_loc)
from ..docstring import docstrings, dedent, safe_modulo
from .logsetup import _get_home


@docstrings.get_sectionsf('safe_list')
@dedent
def safe_list(l):
    """Function to create a list

    Parameters
    ----------
    l: iterable or anything else
        Parameter that shall be converted to a list.

        - If string or any non-iterable, it will be put into a list
        - if iterable, it will be converted to a list

    Returns
    -------
    list
        `l` put (or converted) into a list"""
    if isstring(l):
        return [l]
    try:
        return list(l)
    except TypeError:
        return [l]


class SubDict(UserDict, dict):
    """Class that keeps week reference to the base dictionary

This class is used by the :meth:`RcParams.find_and_replace` method
to provide an easy handable instance that keeps reference to the
base rcParams dictionary."""

    @property
    def data(self):
        """Dictionary representing this :class:`SubDict` instance

        See Also
        --------
        iteritems
        """
        return dict(self.iteritems())

    @property
    def replace(self):
        """:class:`bool`. If True, matching strings in the :attr:`base_str`
        attribute are replaced with an empty string."""
        return self._replace

    @replace.setter
    def replace(self, value):
        def replace_base(key):
            for pattern in self.patterns:
                try:
                    return pattern.match(key).group('key')
                except AttributeError:  # if match is None
                    pass
            raise KeyError(
                "Could not find any matching key for %s in the base "
                "dictionary!" % key)

        value = bool(value)
        if hasattr(self, '_replace') and value == self._replace:
            return
        if not hasattr(self, '_replace'):
            self._replace = value
            return
        # if the value has changed, we change the key in the SubDict instance
        # to match the ones in the base dictionary (if they exist)
        for key, val in DictMethods.iteritems(self):
            try:
                if value:
                    new_key = replace_base(key)
                else:
                    new_key = self._get_val_and_base(key)[0]
            except KeyError:
                continue
            else:
                dict.__setitem__(self, new_key, dict.pop(self, key))
        self._replace = value

    #: :class:`dict`. Reference dictionary
    base = {}

    #: list of strings. The strings that are used to set and get a specific key
    #: from the :attr:`base` dictionary
    base_str = []

    #: list of compiled patterns from the :attr:`base_str` attribute, that
    #: are used to look for the matching keys in :attr:`base`
    patterns = []

    #: :class:`bool`. If True, changes are traced back to the :attr:`base` dict
    trace = False

    @docstrings.get_sectionsf('SubDict.add_base_str')
    @dedent
    def add_base_str(self, base_str, pattern='.+', pattern_base=None,
                     append=True):
        """
        Add further base string to this instance

        Parameters
        ----------
        base_str: str or list of str
            Strings that are used as to look for keys to get and set keys in
            the :attr:`base` dictionary. If a string does not contain
            ``'%(key)s'``, it will be appended at the end. ``'%(key)s'`` will
            be replaced by the specific key for getting and setting an item.
        pattern: str
            Default: ``'.+'``. This is the pattern that is inserted for
            ``%(key)s`` in a base string to look for matches (using the
            :mod:`re` module) in the `base` dictionary. The default `pattern`
            matches everything without white spaces.
        pattern_base: str or list or str
            If None, the whatever is given in the `base_str` is used.
            Those strings will be used for generating the final search
            patterns. You can specify this parameter by yourself to avoid the
            misinterpretation of patterns. For example for a `base_str` like
            ``'my.str'`` it is recommended to additionally provide the
            `pattern_base` keyword with ``'my\.str'``.
            Like for `base_str`, the ``%(key)s`` is appended if not already in
            the string.
        append: bool
            If True, the given `base_str` are appended (i.e. it is first
            looked for them in the :attr:`base` dictionary), otherwise they are
            put at the beginning"""
        base_str = safe_list(base_str)
        pattern_base = safe_list(pattern_base or [])
        for i, s in enumerate(base_str):
            if '%(key)s' not in s:
                base_str[i] += '%(key)s'
        if pattern_base:
            for i, s in enumerate(pattern_base):
                if '%(key)s' not in s:
                    pattern_base[i] += '%(key)s'
        else:
            pattern_base = base_str
        self.base_str = base_str + self.base_str
        self.patterns = list(map(lambda s: re.compile(s.replace(
            '%(key)s', '(?P<key>%s)' % pattern)), pattern_base)) + \
            self.patterns

    docstrings.delete_params('SubDict.add_base_str.parameters', 'append')

    @docstrings.get_sectionsf('SubDict')
    @docstrings.dedent
    def __init__(self, base, base_str, pattern='.+', pattern_base=None,
                 trace=False, replace=True):
        """
        Parameters
        ----------
        base: dict
            base dictionary
        %(SubDict.add_base_str.parameters.no_append)s
        trace: bool
            Default: False. If True, changes in the SubDict are traced back to
            the `base` dictionary. You can change this behaviour also
            afterwards by changing the :attr:`trace` attribute
        replace: bool
            Default: True. If True, everything but the '%%(key)s' part in a
            base string is replaced (see examples below)


        Notes
        -----
        - If a key of matches multiple strings in `base_str`, the first
          matching one is used.
        - the SubDict class is (of course) not that efficient as the
          :attr:`base` dictionary, since we loop multiple times through it's
          keys

        Examples
        --------
        Initialization example:

        .. ipython::

            In [1]: from psyplot import rcParams

            In [2]: d = rcParams.find_and_replace(['plotter.baseplotter.',
               ...:                                'plotter.vector.'])

            In [3]: print d['title']
            Out [3]: None

            In [4]: print d['arrowsize']
            Out [4]: 1.0

        To convert it to a usual dictionary, simply use the :attr:`data`
        attribute

        .. ipython::

            In [5]: d.data

        Note that changing one keyword of your :class:`SubDict` will not change
        the :attr:`base` dictionary, unless you set the :attr:`trace` attribute
        to ``True``.

        .. ipython::

            In [6]: d['title'] = 'my title'

            In [7]: d['title']
            Out [7]: 'my title'

            In [8]: rcParams['plotter.baseplotter.title']
            Out [8]: None

            In [9]: d.trace = True

            In [10]: d['title'] = 'my second title'

            In [11]: d['title']
            Out [11]: 'my second title'

            In [12]: rcParams['plotter.baseplotter.title']
            Out [12]: 'my second title'

        Furthermore, changing the :attr:`replace` attribute will change how you
        can access the keys.

        .. ipython::

            In [13]: d.replace = False

            # now setting d['title'] = 'anything' would raise an error (since
            # d.trace is set to True and 'title' is not a key in the rcParams
            # dictionary. Instead we need
            In [14]: d['plotter.baseplotter.title'] = 'anything'

        See Also
        --------
        RcParams.find_and_replace"""
        self.base = base
        self.base_str = []
        self.patterns = []
        self.replace = bool(replace)
        self.trace = bool(trace)
        self.add_base_str(base_str, pattern=pattern, pattern_base=pattern_base,
                          append=False)

    def __getitem__(self, key):
        if key in DictMethods.iterkeys(self):
            return dict.__getitem__(self, key)
        if not self.replace:
            return self.base[key]
        return self._get_val_and_base(key)[1]

    def __setitem__(self, key, val):
        # set it in the SubDict instance if trace is False
        if not self.trace:
            dict.__setitem__(self, key, val)
            return
        base = self.base
        # set it with the given key, if trace is True
        if not self.replace:
            base[key] = val
            dict.pop(self, key, None)
            return
        # first look if the key already exists in the base dictionary
        for s, patt in self._iter_base_and_pattern(key):
            m = patt.match(s)
            if m and s in base:
                base[m.group()] = val
                return
        # if the key does not exist, we set it
        self.base[key] = val

    def _get_val_and_base(self, key):
        found = False
        for s, patt in self._iter_base_and_pattern(key):
            found = True
            try:
                m = patt.match(s)
                if m:
                    return m.group(), self.base[m.group()]
                else:
                    raise KeyError(
                        "{0} does not match the specified pattern!".format(
                            s))
            except KeyError:
                pass
        if not found:
            raise
        else:
            raise KeyError("{0} does not match the specified pattern!".format(
                            key))

    def _iter_base_and_pattern(self, key):
        return zip(
            map(lambda s: safe_modulo(s, {'key': key}), self.base_str),
            self.patterns)

    def iterkeys(self):
        """Unsorted iterator over keys"""
        patterns = self.patterns
        replace = self.replace
        seen = set()
        for key in six.iterkeys(self.base):
            for pattern in patterns:
                m = pattern.match(key)
                if m:
                    ret = m.group('key') if replace else m.group()
                    if ret not in seen:
                        seen.add(ret)
                        yield ret
                    break
        for key in DictMethods.iterkeys(self):
            if key not in seen:
                yield key

    def iteritems(self):
        """Unsorted iterator over items"""
        return ((key, self[key]) for key in self.iterkeys())

    def itervalues(self):
        """Unsorted iterator over values"""
        return (val for key, val in self.iteritems())

    def update(self, *args, **kwargs):
        """Update the dictionary"""
        for k, v in six.iteritems(dict(*args, **kwargs)):
            self[k] = v

docstrings.delete_params('SubDict.parameters', 'base')


class RcParams(dict):
    """A dictionary object including validation

    validating functions are defined and associated with rc parameters in
    :data:`defaultParams`

    This class is essentially the same as in maplotlibs
    :class:`~matplotlib.RcParams` but has the additional
    :meth:`find_and_replace` method."""

    @property
    def validate(self):
        """Dictionary with validation methods as values"""
        return dict((key, val[1]) for key, val in
                    six.iteritems(defaultParams)
                    if key not in _all_deprecated)

    @property
    def descriptions(self):
        """The description of each keyword in the rcParams dictionary"""
        return {key: val[2] for key, val in six.iteritems(defaultParams)
                if len(val) >= 3}

    HEADER = """Configuration parameters of the psyplot module

You can copy this file (or parts of it) to another path and save it as
PSYPLOTRC. The directory should then be stored in the PSYPLOTCONFIGDIR
environment variable."""

    msg_depr = "%s is deprecated and replaced with %s; please use the latter."
    msg_depr_ignore = "%s is deprecated and ignored. Use %s"

    # validate values on the way in
    def __init__(self, *args, **kwargs):
        for k, v in six.iteritems(dict(*args, **kwargs)):
            try:
                self[k] = v
            except (ValueError, RuntimeError):
                # force the issue
                warn(_rcparam_warn_str.format(key=repr(k), value=repr(v),
                                              func='__init__'))
                dict.__setitem__(self, k, v)

    def __setitem__(self, key, val):
        try:
            if key in _deprecated_map:
                alt_key, alt_val = _deprecated_map[key]
                warn(self.msg_depr % (key, alt_key))
                key = alt_key
                val = alt_val(val)
            elif key in _deprecated_ignore_map:
                alt = _deprecated_ignore_map[key]
                warn(self.msg_depr_ignore % (key, alt))
                return
            try:
                cval = self.validate[key](val)
            except ValueError as ve:
                raise ValueError("Key %s: %s" % (key, str(ve)))
            dict.__setitem__(self, key, cval)
        except KeyError:
            raise KeyError('%s is not a valid rc parameter.\
See rcParams.keys() for a list of valid parameters.' % (key,))

    def __getitem__(self, key):
        if key in _deprecated_map:
            alt_key, alt_val = _deprecated_map[key]
            warn(self.msg_depr % (key, alt_key))
            key = alt_key
        elif key in _deprecated_ignore_map:
            alt = _deprecated_ignore_map[key]
            warn(self.msg_depr_ignore % (key, alt))
            key = alt
        return dict.__getitem__(self, key)

    # the default dict `update` does not use __setitem__
    # so rcParams.update(...) (such as in seaborn) side-steps
    # all of the validation over-ride update to force
    # through __setitem__
    def update(self, *args, **kwargs):
        for k, v in six.iteritems(dict(*args, **kwargs)):
            try:
                self[k] = v
            except (ValueError, RuntimeError):
                # force the issue
                warn(_rcparam_warn_str.format(key=repr(k), value=repr(v),
                                              func='update'))
                dict.__setitem__(self, k, v)

    def __repr__(self):
        import pprint
        class_name = self.__class__.__name__
        indent = len(class_name) + 1
        repr_split = pprint.pformat(dict(self), indent=1,
                                    width=80 - indent).split('\n')
        repr_indented = ('\n' + ' ' * indent).join(repr_split)
        return '{0}({1})'.format(class_name, repr_indented)

    def __str__(self):
        return '\n'.join('{0}: {1}'.format(k, v)
                         for k, v in sorted(self.items()))

    def keys(self):
        """
        Return sorted list of keys.
        """
        k = list(dict.keys(self))
        k.sort()
        return k

    def values(self):
        """
        Return values in order of sorted keys.
        """
        return [self[k] for k in self.keys()]

    def find_all(self, pattern):
        """
        Return the subset of this RcParams dictionary whose keys match,
        using :func:`re.search`, the given ``pattern``.

        Parameters
        ----------
        pattern: str
            pattern as suitable for re.compile

        Returns
        -------
        RcParams
            RcParams instance with entries that match the given `pattern`

        Notes
        -----
        Changes to the returned dictionary are (different from
        :meth:`find_and_replace` are *not* propagated to the parent RcParams
        dictionary.

        See Also
        --------
        find_and_replace"""
        pattern_re = re.compile(pattern)
        return RcParams((key, value)
                        for key, value in self.items()
                        if pattern_re.search(key))

    @docstrings.dedent
    def find_and_replace(self, *args, **kwargs):
        """
        Like :meth:`find_all` but the given strings are replaced

        This method returns a dictionary-like object that keeps weak reference
        to this rcParams instance. The resulting `SubDict` instance takes the
        keys from this rcParams instance but leaves away what is found in
        `base_str`.

        ``*args`` and ``**kwargs`` are determined by the :class:`SubDict`
        class, where the `base` dictionary is this one.

        Parameters
        ----------
        %(SubDict.parameters.no_base)s

        Returns
        -------
        SubDict
            SubDict with this rcParams instance as reference.

        Examples
        --------
        .. ipython::

            In [1]: from psyplot import rcParams

            In [2]: d = rcParams.find_and_replace(['plotter.baseplotter.',
               ...:                                'plotter.vector.'])

            In [3]: print d['title']
            Out [3]: None

            In [4]: print d['arrowsize']
            Out [4]: 1.0

        See Also
        --------
        find_all
        SubDict"""
        return SubDict(self, *args, **kwargs)

    def load_from_file(self, fname=None):
        """Update rcParams from user-defined settings

        This function updates the instance with what is found in `fname`

        Parameters
        ----------
        fname: str
            Path to the yaml configuration file. Possible keys of the
            dictionary are defined by :data:`config.rcsetup.defaultParams`.
            If None, the :func:`config.rcsetup.psyplot_fname` function is used.

        See Also
        --------
        dump_to_file, psyplot_fname"""
        fname = fname or psyplot_fname()
        if fname and os.path.exists(fname):
            with open(fname) as f:
                self.update(yaml.load(f))

    def dump(self, fname=None, overwrite=True, include_keys=None,
             exclude_keys=['project.plotters'], include_descriptions=True):
        """Dump this instance to a yaml file

        Parameters
        ----------
        fname: str or None
            file name to write to. If Non, the string that would be written
            to a file is returned
        overwrite: bool
            If True and `fname` already exists, it will be overwritten
        include_keys: None or list of str
            Keys in the dictionary to be included. If None, all keys are
            included
        exclude_keys: list of str
            Keys from the :class:`RcParams` instance to be excluded

        Returns
        -------
        str or None
            if fname is ``None``, the string is returned. Otherwise, ``None``
            is returned

        Raises
        ------
        IOError
            If `fname` already exists and `overwrite` is False

        See Also
        --------
        load_from_file"""
        if fname is not None and not overwrite and os.path.exists(fname):
            raise IOError(
                '%s already exists! Set overwrite=True to overwrite it!' % (
                    fname))
        kwargs = dict(encoding='utf-8') if six.PY2 else {}
        d = {key: val for key, val in six.iteritems(self) if (
                include_keys is None or key in include_keys) and
             key not in exclude_keys}
        if include_descriptions:
            s = yaml.dump(d, **kwargs)
            desc = self.descriptions
            i = 2
            lines = ['\n'.join('# ' + l for l in self.HEADER.split('\n')),
                     '\nCreated with python\n%s\n\n' % sys.version] + s.split(
                         '\n')
            for l in lines[2:]:
                key = l.split(':')[0]
                if key in desc:
                    lines.insert(i, '# ' + desc[key])
                    i += 1
                i += 1
            s = '\n'.join(lines)
            if fname is None:
                return s
            else:
                with open(fname, 'w') as f:
                    f.write(s)
        else:
            if fname is None:
                return yaml.dump(d, **kwargs)
            with open(fname, 'w') as f:
                yaml.dump(d, f, **kwargs)
        return None


def psyplot_fname():
    """
    Get the location of the config file.

    The file location is determined in the following order

    - `$PWD/psyplotrc.yaml`

    - environment variable `PSYPLOTRC` (pointing to the file location or a
      directory containing the file `psyplotrc.yaml`)

    - `$PSYPLOTCONFIGDIR/psyplot`

    - On Linux,

          - `$HOME/.config/psyplot/psyplotrc.yaml`

    - On other platforms,

         - `$HOME/.psyplot/psyplotrc.yaml` if `$HOME` is defined.

    - Lastly, it looks in `$PSYPLOTDATA/psyplotrc.yaml` for a
      system-defined copy.

    Returns
    -------
    None or str
        None, if no file could be found, else the path to the psyplot
        configuration file

    Notes
    -----
    This function is taken from the matplotlib [1] module

    References
    ----------
    [1]: http://matplotlib.org/api/"""
    cwd = getcwd()
    fname = os.path.join(cwd, 'psyplotrc.yaml')
    if os.path.exists(fname):
        return fname

    if 'PSYPLOTRC' in os.environ:
        path = os.environ['PSYPLOTRC']
        if os.path.exists(path):
            if os.path.isdir(path):
                fname = os.path.join(path, 'psyplotrc.yaml')
                if os.path.exists(fname):
                    return fname
            else:
                return path

    configdir = _get_configdir()
    if configdir is not None:
        fname = os.path.join(configdir, 'psyplotrc.yaml')
        if os.path.exists(fname):
            return fname

    return None


def _get_configdir():
    """
    Return the string representing the configuration directory.

    The directory is chosen as follows:

    1. If the MPLCONFIGDIR environment variable is supplied, choose that.

    2a. On Linux, choose `$HOME/.config`.

    2b. On other platforms, choose `$HOME/.matplotlib`.

    3. If the chosen directory exists, use that as the
       configuration directory.
    4. A directory: return None.

    Notes
    -----
    This function is taken from the matplotlib [1] module

    References
    ----------
    [1]: http://matplotlib.org/api/"""
    configdir = os.environ.get('PSYPLOTCONFIGDIR')
    if configdir is not None:
        return os.path.abspath(configdir)

    p = None
    h = _get_home()
    if (sys.platform.startswith('linux') and h is not None):
        p = os.path.join(h, '.config/psyplot')
    elif h is not None:
        p = os.path.join(h, '.psyplot')

    return p


def try_and_error(*funcs):
    """Apply multiple validation functions

    Parameters
    ----------
    ``*funcs``
        Validation functions to test

    Returns
    -------
    function"""
    def validate(value):
        for func in funcs:
            try:
                return func(value)
            except (ValueError, TypeError):

                continue
        raise
    return validate


def validate_none(b):
    """Validate that None is given

    Parameters
    ----------
    b: {None, 'none'}
        None or string (the case is ignored)

    Returns
    -------
    None

    Raises
    ------
    ValueError"""
    if isinstance(b, six.string_types):
        b = b.lower()
    if b is None or b == 'none':
        return None
    else:
        raise ValueError('Could not convert "%s" to None' % b)


def validate_axiscolor(value):
    """Validate a dictionary containing axiscolor definitions

    Parameters
    ----------
    value: dict
        see :attr:`psyplot.plotter.baseplotter.axiscolor`

    Returns
    -------
    dict

    Raises
    ------
    ValueError"""
    validate = try_and_error(validate_none, validate_color)
    possible_keys = {'right', 'left', 'top', 'bottom'}
    try:
        value = dict(value)
        false_keys = set(value) - possible_keys
        if false_keys:
            raise ValueError("Wrong keys (%s)!" % (', '.join(false_keys)))
        for key, val in value.items():
            value[key] = validate(val)
    except:
        value = dict(zip(possible_keys, repeat(validate(value))))
    return value


def validate_text(value):
    """Validate a text formatoption

    Parameters
    ----------
    value: see :attr:`psyplot.plotter.labelplotter.text`

    Raises
    ------
    ValueError"""
    possible_transform = ['axes', 'fig', 'data']
    validate_transform = ValidateInStrings('transform', possible_transform,
                                           True)
    tests = [validate_float, validate_float, validate_str,
             validate_transform, dict]
    if isinstance(value, six.string_types):
        try:
            from psyplot import rcParams
        except ImportError:
            rcParams = defaultParams
        xpos, ypos = rcParams['texts.default_position']
        return [(xpos, ypos, value, 'axes', {'ha': 'right'})]
    elif isinstance(value, tuple):
        value = [value]
    try:
        value = list(value)[:]
    except TypeError:
        raise ValueError("Value must be string or list of tuples!")
    for i, val in enumerate(value):
        try:
            val = tuple(val)
        except TypeError:
            raise ValueError(
                "Text must be an iterable of the form "
                "(x, y, s[, trans, params])!")
        if len(val) < 3:
            raise ValueError(
                "Text tuple must at least be like [x, y, s], with floats x, "
                "y and string s!")
        elif len(val) == 3 or isinstance(val[3], dict):
            val = list(val)
            val.insert(3, 'data')
            if len(val) == 4:
                val += [{}]
            val = tuple(val)
        if len(val) > 5:
            raise ValueError(
                "Text tuple must not be longer then length 5. It can be "
                "like (x, y, s[, trans, params])!")
        value[i] = (validate(x) for validate, x in zip(tests, val))
    return value


validate_scale = ValidateInStrings('scale', ['logx', 'logy', 'logxy'], True)


class validate_list(object):
    """Validate a list of the specified `dtype`

    Parameters
    ----------
    dtype: object
        A datatype (e.g. :class:`float`) that shall be used for the conversion
    """

    def __init__(self, dtype=None):
        """Initialization function"""
        #: data type (e.g. :class:`float`) used for the conversion
        self.dtype = dtype

    def __call__(self, l):
        """Validate whether `l` is a list with contents of :attr:`dtype`

        Parameters
        ----------
        l: list-like

        Returns
        -------
        list
            list with values of dtype :attr:`dtype`

        Raises
        ------
        ValueError"""
        try:
            if self.dtype is None:
                l = list(l)
            else:
                l = list(map(self.dtype, l))
        except TypeError:
            if self.dtype is None:
                raise ValueError(
                    "Could not convert to list!")
            else:
                raise ValueError(
                    "Could not convert to list of type %s!" % str(self.dtype))
        return l


def validate_str(s):
    """Validate a string

    Parameters
    ----------
    s: str

    Returns
    -------
    str

    Raises
    ------
    ValueError"""
    if not isinstance(s, six.string_types):
        raise ValueError("Did not found string!")
    return six.text_type(s)


def validate_dict(d):
    """Validate a dictionary

    Parameters
    ----------
    d: dict or str
        If str, it must be a path to a yaml file

    Returns
    -------
    dict

    Raises
    ------
    ValueError"""
    try:
        return dict(d)
    except TypeError:
        try:
            d = validate_path_exists(d)
            return dict(yaml.load(d))
        except:
            raise ValueError("Could not convert to dictionary!")


def validate_cbarpos(value):
    """Validate a colorbar position

    Parameters
    ----------
    value: bool or str
        A string can be a combination of 'sh|sv|fl|fr|ft|fb|b|r'

    Returns
    -------
    list
        list of strings with possible colorbar positions

    Raises
    ------
    ValueError"""
    patt = 'sh|sv|fl|fr|ft|fb|b|r'
    if value is True:
        value = {'b'}
    elif not value:
        value = set()
    elif isinstance(value, six.string_types):
        for s in re.finditer('[^%s]+' % patt, value):
            warn("Unknown colorbar position %s!" % s.group())
        value = set(re.findall(patt, value))
    else:
        value = validate_stringset(value)
        for s in (s for s in value
                  if not re.match(patt, s)):
            warn("Unknown colorbar position %s!" % s)
            value.remove(s)
    return value


def validate_cmap(val):
    """Validate a colormap

    Parameters
    ----------
    val: str or :class:`mpl.colors.Colormap`

    Returns
    -------
    str or :class:`mpl.colors.Colormap`

    Raises
    ------
    ValueError"""
    from matplotlib.colors import Colormap
    try:
        return validate_str(val)
    except ValueError:
        if not isinstance(val, Colormap):
            raise ValueError(
                "Could not find a valid colormap!")
        return val


def validate_cmaps(cmaps):
    """Validate a dictionary of color lists

    Parameters
    ----------
    cmaps: dict
        a mapping from a colormap name to a list of colors

    Raises
    ------
    ValueError
        If one of the values in `cmaps` is not a color list

    Notes
    -----
    For all items (listname, list) in `cmaps`, the reversed list is
    automatically inserted with the ``listname + '_r'`` key."""
    cmaps = {validate_str(key): validate_colorlist(val) for key, val in cmaps}
    for key, val in six.iteritems(cmaps):
        cmaps.setdefault(key + '_r', val[::-1])
    return cmaps


def validate_stringlist(s):
    """Validate a list of strings

    Parameters
    ----------
    val: iterable of strings

    Returns
    -------
    list
        list of str

    Raises
    ------
    ValueError"""
    if isinstance(s, six.string_types):
        return [six.text_type(v.strip()) for v in s.split(',') if v.strip()]
    else:
        try:
            return [six.text_type(v) for v in s if v]
        except TypeError as e:
            raise ValueError(e.message)

validate_extend = ValidateInStrings('extend',
                                    ['neither', 'both', 'min', 'max'])


def validate_norm(val):
    """Validate a normalization

    Parameters
    ----------
    val: str, None or :class:`mpl.colors.Normalize` instance
        see :attr:`plotter.Plot2D.norm`

    Returns
    -------
    list
        list that can be used for :attr:`psyplot.plotter.Plot2D.norm`

    Raises
    ------
    ValueError"""
    if val == 'bounds':
        return val
    try:
        return validate_none(val)
    except ValueError:
        pass
    if not isinstance(val, mpl.colors.Normalize):
        raise ValueError(
            "Wrong value of norm! Can be either 'bounds', None or a "
            "matplotlib.colors.Normalize instance!")
    return val


def validate_stringset(*args, **kwargs):
    """Validate a set of strings

    Parameters
    ----------
    val: iterable of strings

    Returns
    -------
    set
        set of str

    Raises
    ------
    ValueError"""
    return set(validate_stringlist(*args, **kwargs))


def validate_opacity(val):
    """Validate a valid opacity value

    Parameters
    ----------
    val: float or array
        see :attr:`psyplot.plotter.Plot2D.opacity`

    Returns
    -------
    float or array

    Raises
    ------
    ValueError"""
    from numpy import array

    def validate_alpha(v):
        if v < 0 or v > 1:
            raise ValueError(
                "Opacity float must not be smaller than 0 or greater than 1!")
        return v
    try:
        return validate_alpha(validate_float(val))
    except ValueError:
        pass
    val = array(val).astype(float)
    if val.ndim == 0 or val.ndim > 2:
        raise ValueError("Opacity arrays can only be one or two-dimensional!")
    if val.ndim == 1:
        if len(val.ndim) < 2:
            raise ValueError(
                "Need at least two values for one dimensional opacity array!")
        map(validate_alpha, val)
    else:
        if val.shape[1] == 2:
            raise ValueError(
                "2-dimensional opacity arrays must be of shape (N, 2)! Found "
                "{0}.".format(val.shape))
        map(validate_alpha, val[:, 1])
    return val


def validate_shape_exists(p):
    """Validate that a shapefile exists

    Parameters
    ----------
    p: str
        path to a shapefile suitable for a :class:`shapefile.Reader` instance

    Returns
    -------
    str: the given input

    Raises
    ------
    ValueError"""
    if p is None:
        return p
    from shapefile import Reader
    try:
        Reader(p)
    except:
        raise ValueError("Could not open shapefile with {0}!".format(p))
    return p


def validate_float(s):
    """convert `s` to float or raise

    Returns
    -------
    s converted to a float: float

    Raises
    ------
    ValueError"""
    try:
        return float(s)
    except (ValueError, TypeError):
        raise ValueError('Could not convert "%s" to float' % s)


def validate_iter(value):
    """Validate that the given value is an iterable"""
    try:
        return iter(value)
    except TypeError:
        raise ValueError("%s is not an iterable!" % repr(value))


def validate_fontweight(value):
    if value is None:
        return None
    elif isinstance(value, six.string_types):
        return six.text_type(value)
    elif mpl_version >= 1.5:
        return validate_float(value)
    raise ValueError("Font weights must be None or a string!")


def validate_limits(value):
    if value is None or isinstance(value, six.string_types):
        return (value, value)
    if not len(value) == 2:
        raise ValueError("Limits must have length 2!")
    return value


class DictValValidator(object):

    def __init__(self, key, valid, validators, default, ignorecase=False):
        self.key = key
        self.valid = valid
        self.key_validator = ValidateInStrings(key, valid, ignorecase)
        self.default = default
        self.validate = validators

    def __call__(self, value):
        if isinstance(value, dict) and value and all(
                isinstance(key, six.string_types) for key in value):
            failed_key = False
            for key, val in six.iteritems(value):
                try:
                    new_key = self.key_validator(key)
                except ValueError:
                    failed_key = True
                    break
                else:
                    value[new_key] = self.validate(value.pop(key))
            if failed_key:
                if self.default is None:
                    value = self.validate(value)
                    value = dict(zip(self.valid, repeat(value)))
                else:
                    value = {self.default: self.validate(value)}
        elif self.default is None:
            value = self.validate(value)
            value = dict(zip(self.valid, repeat(value)))
        else:
            value = {self.default: self.validate(value)}
        return value


class TicksValidator(ValidateInStrings):

    def __call__(self, val):
        # validate the ticks
        # if None, int or tuple (defining min- and max-range), pass
        if val is None or isinstance(val, int) or (
                isinstance(val, tuple) and len(val) <= 3):
            return val
        # strings must be in the given list
        elif isinstance(val, six.string_types):
            return [ValidateInStrings.__call__(self, val), None]
        elif isinstance(val[0], six.string_types):
            return [ValidateInStrings.__call__(self, val[0])] + list(val[1:])
        # otherwise we assume an array
        else:
            return validate_list()(val)


class BoundsValidator(ValidateInStrings):

    def __init__(self, *args, **kwargs):
        """
        For parameter description see
        :class:`matplotlib.rcsetup.ValidateInStrings`.

        Other Parameters
        ----------------
        inis: tuple
            Tuple of object types that may pass the check
        default: str
            The default string to use for an integer (Default: 'rounded')"""
        self.possible_instances = kwargs.pop('inis', None)
        self.default = kwargs.pop('default', 'rounded')
        ValidateInStrings.__init__(self, *args, **kwargs)

    def instance_check(self, val):
        if self.possible_instances:
            return isinstance(val, self.possible_instances)
        return False

    def __call__(self, val):
        if val is None or self.instance_check(val):
            return val
        elif isinstance(val, int):
            return [self.default, val]
        elif isinstance(val, six.string_types):
            return [ValidateInStrings.__call__(self, val), None]
        elif isinstance(val[0], six.string_types):
            return [ValidateInStrings.__call__(self, val[0])] + list(val[1:])
        # otherwise we assume an array
        else:
            return validate_list(float)(val)


class ProjectionValidator(ValidateInStrings):

    def __call__(self, val):
        if isinstance(val, six.string_types):
            return ValidateInStrings.__call__(self, val)
        return val  # otherwise we skip the validation


class LineWidthValidator(ValidateInStrings):

    def __call__(self, val):
        if val is None:
            return val
        elif isinstance(val, six.string_types):
            return [ValidateInStrings.__call__(self, val), 1.0]
        elif asarray(val).ndim and isinstance(val[0], six.string_types):
            return [ValidateInStrings.__call__(self, val[0])] + list(val[1:])
        # otherwise we assume an array
        else:
            return asarray(val, float)


validate_ticklabels = try_and_error(validate_none, validate_str,
                                    validate_stringlist)


def validate_dict_yaml(s):
    if isinstance(s, dict):
        return s
    validate_path_exists(s)
    if s is not None:
        with open(s) as f:
            return yaml.load(f)


bound_strings = ['data', 'mid', 'rounded', 'roundedsym', 'minmax', 'sym']

tick_strings = bound_strings + ['hour', 'day', 'week', 'month', 'monthend',
                                'monthbegin', 'year', 'yearend', 'yearbegin']


#: :class:`dict` with default values and validation functions
defaultParams = {
    # BasePlot
    'plotter.baseplotter.tight': [False, validate_bool,
                                  'fmt key for tight layout of the plots'],
    'plotter.simpleplotter.grid': [
        False, try_and_error(validate_bool_maybe_none, validate_color),
        'fmt key to visualize the grid on simple plots (i.e. without '
        'projection)'],
    # labels
    'plotter.baseplotter.title': [
        '', six.text_type, 'fmt key to control the title of the axes'],
    'plotter.baseplotter.figtitle': [
        '', six.text_type, 'fmt key to control the title of the axes'],
    'plotter.baseplotter.text': [
        [], validate_text, 'fmt key to show text anywhere on the plot'],
    'plotter.simpleplotter.ylabel': [
        '', six.text_type, 'fmt key to modify the y-axis label for simple'
        'plot (i.e. plots withouth projection)'],
    'plotter.simpleplotter.xlabel': [
        '', six.text_type, 'fmt key to modify the y-axis label for simple'
        'plot (i.e. plots withouth projection)'],
    'plotter.plot2d.clabel': [
        '', six.text_type, 'fmt key to modify the colorbar label for 2D'
        'plots'],
    # text sizes
    'plotter.baseplotter.titlesize': [
        'large', validate_fontsize,
        'fmt key for the fontsize of the axes title'],
    'plotter.baseplotter.figtitlesize': [
        12, validate_fontsize, 'fmt key for the fontsize of the figure title'],
    'plotter.simpleplotter.labelsize': [
        'medium', DictValValidator(
            'labelsize', ['x', 'y'], validate_fontsize, None, True),
        'fmt key for the fontsize of the x- and y-l abel of simple plots '
        '(i.e. without projection)'],
    'plotter.simpleplotter.ticksize': [
        'medium', DictValValidator(
            'ticksize', ['major', 'minor'], validate_fontsize, 'major', True),
        'fmt key for the fontsize of the ticklabels of x- and y-axis of '
        'simple plots (i.e. without projection)'],
    'plotter.plot2d.cticksize': [
        'medium', validate_fontsize,
        'fmt key for the fontsize of the ticklabels of the colorbar of 2D '
        'plots'],
    'plotter.plot2d.clabelsize': [
        'medium', validate_fontsize,
        'fmt key for the fontsize of the colorbar label'],
    # text weights
    'plotter.baseplotter.titleweight': [
        None, validate_fontweight,
        'fmt key for the fontweight of the axes title'],
    'plotter.baseplotter.figtitleweight': [
        None, validate_fontweight,
        'fmt key for the fontweight of the figure title'],
    'plotter.simpleplotter.labelweight': [
        None, DictValValidator(
        'labelweight', ['x', 'y'], validate_fontweight, None, True),
        'fmt key for the fontweight of the x- and y-l abel of simple plots '
        '(i.e. without projection)'],
    'plotter.simpleplotter.tickweight': [None, DictValValidator(
        'tickweight', ['major', 'minor'], validate_fontweight, 'major', True),
        'fmt key for the fontweight of the ticklabels of x- and y-axis of '
        'simple plots (i.e. without projection)'],
    'plotter.plot2d.ctickweight': [
        None, validate_fontweight,
        'fmt key for the fontweight of the ticklabels of the colorbar of 2D '
        'plots'],
    'plotter.plot2d.clabelweight': [
        None, validate_fontweight,
        'fmt key for the fontweight of the colorbar label'],
    # text properties
    'plotter.baseplotter.titleprops': [{}, validate_dict],
    'plotter.baseplotter.figtitleprops': [{}, validate_dict],
    'plotter.simpleplotter.labelprops': [{}, DictValValidator(
        'labelprops', ['x', 'y'], validate_dict, None, True)],
    'plotter.simpleplotter.xtickprops': [
        {'major': {}, 'minor': {}}, DictValValidator(
            'xtickprops', ['major', 'minor'], validate_dict, 'major', True)],
    'plotter.simpleplotter.ytickprops': [
        {'major': {}, 'minor': {}}, DictValValidator(
            'ytickprops', ['major', 'minor'], validate_dict, 'major', True)],
    'plotter.plot2d.clabelprops': [{}, validate_dict],
    'plotter.plot2d.ctickprops': [{}, validate_dict],
    # axis color
    'plotter.baseplotter.axiscolor': [None, validate_axiscolor],

    # SimplePlot
    'plotter.simpleplotter.plot': ['line', try_and_error(
        validate_none, ValidateInStrings(
            '1d plot', ['line', 'bar', 'violin'], True))],
    'plotter.simpleplotter.transpose': [False, validate_bool],
    'plotter.simpleplotter.color': [None, try_and_error(
        validate_none, validate_cmap, validate_iter)],
    'plotter.simpleplotter.ylim': ['rounded', validate_limits],
    'plotter.simpleplotter.xlim': ['rounded', validate_limits],
    'plotter.simpleplotter.scale': [None, try_and_error(
        validate_none, validate_scale)],
    'plotter.simpleplotter.xticks': [
        {'major': None, 'minor': None}, DictValValidator(
            'xticks', ['major', 'minor'], TicksValidator(
                'xticks', tick_strings, True), 'major', True)],
    'plotter.simpleplotter.yticks': [
        {'major': None, 'minor': None}, DictValValidator(
            'yticks', ['major', 'minor'], TicksValidator(
                'yticks', tick_strings, True), 'major', True)],
    'plotter.simpleplotter.xticklabels': [None, DictValValidator(
        'xticklabels', ['major', 'minor'], validate_ticklabels, 'major',
        True)],
    'plotter.simpleplotter.yticklabels': [None, DictValValidator(
        'yticklabels', ['major', 'minor'], validate_ticklabels, 'major',
        True)],
    'plotter.simpleplotter.xrotation': [0, validate_float],
    'plotter.simpleplotter.yrotation': [0, validate_float],
    'plotter.simpleplotter.legendlabels': ['%(arr_name)s', try_and_error(
        validate_str, validate_list(str))],
    'plotter.simpleplotter.legend': [True, try_and_error(
        validate_bool, validate_int, validate_dict, validate_legend_loc)],

    # Plot2D
    'plotter.plot2d.plot': ['mesh', try_and_error(
        validate_none, ValidateInStrings('2d plot', ['mesh', 'tri'], True))],
    'plotter.plot2d.cbar': [['b'], validate_cbarpos],
    'plotter.plot2d.cbarspacing': ['uniform', validate_str],
    'plotter.plot2d.miss_color': [None, try_and_error(validate_none,
                                                      validate_color)],
    'plotter.plot2d.cmap': ['white_blue_red', validate_cmap],
    'plotter.plot2d.cticks': [None, try_and_error(
        validate_none, BoundsValidator(
            'bounds', ['bounds'] + bound_strings, True, default='bounds'))],
    'plotter.plot2d.cticklabels': [None, validate_ticklabels],
    'plotter.plot2d.extend': ['neither', validate_extend],
    'plotter.plot2d.rasterized': [True, validate_bool],
    'plotter.plot2d.bounds': ['rounded', BoundsValidator(
        'bounds', bound_strings, True, inis=mpl.colors.Normalize)],
    'plotter.plot2d.norm': ['bounds', validate_norm],
    'plotter.plot2d.opacity': [None, try_and_error(validate_none,
                                                   validate_opacity)],
    'plotter.plot2d.datagrid': [None, try_and_error(
        validate_none, validate_dict, validate_str)],

    # MapBase
    'plotter.maps.latlon': [True, validate_bool],
    'plotter.maps.lonlatbox': [None, try_and_error(
        validate_none, validate_str, validate_nseq_float(4))],
    'plotter.maps.map_extent': [None, try_and_error(
        validate_none, validate_str, validate_nseq_float(4))],
    'plotter.maps.clon': [None, try_and_error(
        validate_none, validate_float, validate_str)],
    'plotter.maps.clat': [None, try_and_error(
        validate_none, validate_float, validate_str)],
    'plotter.maps.lineshapes': [None, try_and_error(
        validate_none, validate_dict, validate_str, validate_stringlist)],
    'plotter.maps.grid_labels': [True, validate_bool],
    'plotter.maps.grid_labelsize': [12.0, validate_fontsize],
    'plotter.maps.grid_color': ['k', try_and_error(validate_none,
                                                   validate_color)],
    'plotter.maps.grid_settings': [{}, validate_dict],
    'plotter.maps.xgrid': [True, try_and_error(
        validate_bool_maybe_none, BoundsValidator('bounds', bound_strings,
                                                  True))],
    'plotter.maps.ygrid': [True, try_and_error(
        validate_bool_maybe_none, BoundsValidator('bounds', bound_strings,
                                                  True))],
    'plotter.maps.projection': ['cyl', ProjectionValidator(
        'projection', ['northpole', 'ortho', 'southpole', 'moll', 'geo',
                       'robin', 'cyl'], True)],
    'plotter.maps.transform': ['cyl', ProjectionValidator(
        'projection', ['northpole', 'ortho', 'southpole', 'moll', 'geo',
                       'robin', 'cyl'], True)],
    'plotter.maps.plot.min_circle_ratio': [0.05, validate_float],
    'plotter.maps.lsm': [True, try_and_error(validate_bool,
                                             validate_float)],
    'plotter.maps.mask': [None, lambda x: x],  # TODO: implement validation

    'plotter.baseplotter.maskleq': [None, try_and_error(
        validate_none, validate_float)],
    'plotter.baseplotter.maskless': [None, try_and_error(
        validate_none, validate_float)],
    'plotter.baseplotter.maskgreater': [None, try_and_error(
        validate_none, validate_float)],
    'plotter.baseplotter.maskgeq': [None, try_and_error(
        validate_none, validate_float)],
    'plotter.baseplotter.maskbetween': [None, try_and_error(
        validate_none, validate_nseq_float(2))],

    # WindPlot
    'plotter.vector.plot': ['quiver', try_and_error(
        validate_none, ValidateInStrings(
            '2d plot', ['quiver', 'stream'], True))],
    'plotter.vector.arrowsize': [None, try_and_error(
        validate_none, validate_float)],
    'plotter.vector.arrowstyle': ['-|>', ValidateInStrings(
        'arrowstyle', ArrowStyle._style_list)],
    'plotter.vector.density': [1.0, try_and_error(
        validate_float, validate_list(float))],
    'plotter.vector.linewidth': [None, LineWidthValidator(
            'linewidth', ['absolute', 'u', 'v'], True)],
    'plotter.vector.color': ['k', try_and_error(
        validate_float, validate_color, ValidateInStrings(
            'color', ['absolute', 'u', 'v'], True))],
    'plotter.vector.reduceabove': [None, try_and_error(
        validate_none, validate_nseq_float(2))],
    'plotter.vector.lengthscale': ['lin', ValidateInStrings(
        'lengthscale', ['lin', 'log'], True)],

    # user defined plotter keys
    'plotter.user': [{}, validate_dict_yaml],

    # decoder
    'decoder.x': [set(), validate_stringset,
                  'names that shall be interpreted as the longitudinal x dim'],
    'decoder.y': [set(), validate_stringset,
                  'names that shall be interpreted as the latitudinal y dim'],
    'decoder.z': [set(), validate_stringset,
                  'names that shall be interpreted as the vertical z dim'],
    'decoder.t': [{'time'}, validate_stringset,
                  'names that shall be interpreted as the time dimension'],
    'decoder.interp_kind': [
        'linear', validate_str,
        'interpolation method to calculate 2D-bounds (see the `kind` parameter'
        'in the :meth:`psyplot.data.CFDecoder.get_plotbounds` method)'],

    # data
    'datapath': [None, validate_path_exists, 'path for supplementary data'],

    # default texts
    'texts.labels': [{'tinfo': '%H:%M',
                      'dtinfo': '%B %d, %Y. %H:%M',
                      'dinfo': '%B %d, %Y',
                      'desc': '%(long_name)s [%(units)s]',
                      'sdesc': '%(name)s [%(units)s]'}, validate_dict,
                     'labels that shall be replaced in TextBase formatoptions',
                     ' (e.g. the title formatoption) when inserted within '
                     'curly braces ({}))'],
    'texts.default_position': [(1., 1.), validate_nseq_float(2),
                               'default position for the text fmt key'],
    'texts.delimiter': [', ', validate_str,
                        'default delimiter to separate netCDF meta attributes '
                        'when displayed on the plot'],
    'ticks.which': ['major', ValidateInStrings(
        'ticks.which', ['major', 'minor'], True),
        'default tick that is used when using a x- or y-tick formatoption'],

    # color lists for user-defined colormaps (see for example
    # psyplot.plotter.colors._cmapnames)
    'colors.cmaps': [
        {}, validate_cmaps,
        'User defined color lists that shall be accessible through the '
        ':meth:`psyplot.plotter.colors.get_cmap` function'],

    # yaml file that holds definitions of lonlatboxes
    'lonlatbox.boxes': [
        {}, validate_dict_yaml,
        'longitude-latitude boxes that shall be accessible for the lonlatbox, '
        'map_extent, etc. keywords. May be a dictionary or the path to a '
        'yaml file'],

    # list settings
    'lists.auto_update': [True, validate_bool,
                          'default value (boolean) for the auto_update '
                          'parameter in the initialization of Plotter, '
                          'Project, etc. instances'],

    # project settings
    # auto_import: If True the plotters in project,plotters are automatically
    # imported
    'project.auto_import': [False, validate_bool,
                            'boolean controlling whether all plotters '
                            'specified in the project.plotters item will be '
                            'automatically imported when importing the '
                            'psyplot.project module'],
    'project.plotters': [{  # these plotters are automatically registered
        'plot1d': {
            'module': 'psyplot.plotter.simpleplotter',
            'plotter_name': 'SimplePlotter',
            'prefer_list': True,
            'default_slice': None,
            'summary': 'Make a simple plot of one-dimensional data'},
        'plot2d': {
            'module': 'psyplot.plotter.simpleplotter',
            'plotter_name': 'Simple2DPlotter',
            'prefer_list': False,
            'default_slice': 0,
            'default_dims': {'x': slice(None), 'y': slice(None)},
            'summary': 'Make a simple plot of a 2D scalar field'},
        'vector': {
            'module': 'psyplot.plotter.simpleplotter',
            'plotter_name': 'SimpleVectorPlotter',
            'prefer_list': False,
            'default_slice': 0,
            'default_dims': {'x': slice(None), 'y': slice(None)},
            'summary': 'Make a simple plot of a 2D vector field',
            'example_call': "filename, name=[['u_var', 'v_var']], ..."},
        'maps': {
            'module': 'psyplot.plotter.maps',
            'plotter_name': 'MapPlotter',
            'plot_func': False},
        'mapplot': {
            'module': 'psyplot.plotter.maps',
            'plotter_name': 'FieldPlotter',
            'prefer_list': False,
            'default_slice': 0,
            'default_dims': {'x': slice(None), 'y': slice(None)},
            'summary': 'Plot a 2D scalar field on a map'},
        'mapvector': {
            'module': 'psyplot.plotter.maps',
            'plotter_name': 'VectorPlotter',
            'prefer_list': False,
            'default_slice': 0,
            'default_dims': {'x': slice(None), 'y': slice(None)},
            'summary': 'Plot a 2D vector field on a map',
            'example_call': "filename, name=[['u_var', 'v_var']], ..."},
        'mapcombined': {
            'module': 'psyplot.plotter.maps',
            'plotter_name': 'CombinedPlotter',
            'prefer_list': True,
            'default_slice': 0,
            'default_dims': {'x': slice(None), 'y': slice(None)},
            'summary': ('Plot a 2D scalar field with an overlying vector field'
                        'on a map'),
            'example_call': (
                "filename, name=[['my_variable', ['u_var', 'v_var']]], ...")},
        }, validate_dict,
        'mapping from identifier to plotter definitions for the Project class.'
        ' See the :func:`psyplot.project.register_plotter` function for '
        'possible keywords and values'],
    }

# add combinedplotter strings for windplot
_subd = SubDict(defaultParams, ['plotter.vector.', 'plotter.plot2d.'])
for _key in ['plot', 'cbar', 'cmap', 'bounds', 'cticksize', 'cbarspacing',
             'ctickweight', 'ctickprops', 'clabel', 'cticks', 'cticklabels',
             'clabelsize', 'clabelprops', 'clabelweight']:
    defaultParams['plotter.combinedsimple.v%s' % _key] = _subd[_key]
defaultParams['plotter.combinedsimple.plot'] = defaultParams[
    'plotter.plot2d.plot']
del _key, _subd


_all_deprecated = []
_deprecated_map = {}
_deprecated_ignore_map = {}


_rcparam_warn_str = ("Trying to set {key} to {value} via the {func} "
                     "method of RcParams which does not validate cleanly. ")

#: :class:`~psyplot.config.rcsetup.RcParams` instance that stores default
#: formatoptions and configuration settings.
rcParams = RcParams(**{key: val[0] for key, val in defaultParams.items()})
rcParams.load_from_file()
