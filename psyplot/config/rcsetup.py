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
from itertools import chain
from collections import defaultdict
from psyplot.warning import warn
from psyplot.compat.pycompat import (
    UserDict, DictMethods, getcwd, zip, isstring, map)
from psyplot.docstring import docstrings, dedent, safe_modulo, dedents
from psyplot.config.logsetup import _get_home


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
        Initialization example::

            >>> from psyplot import rcParams
            >>> d = rcParams.find_and_replace(['plotter.baseplotter.',
            ...                                'plotter.vector.'])
            >>> print d['title']

            >>> print d['arrowsize']
            1.0

        To convert it to a usual dictionary, simply use the :attr:`data`
        attribute::

            >>> d.data
            {'title': None, 'arrowsize': 1.0, ...}

        Note that changing one keyword of your :class:`SubDict` will not change
        the :attr:`base` dictionary, unless you set the :attr:`trace` attribute
        to ``True``::

            >>> d['title'] = 'my title'
            >>> print(d['title'])
            my title

            >>> print(rcParams['plotter.baseplotter.title'])

            >>> d.trace = True
            >>> d['title'] = 'my second title'
            >>> print(d['title'])
            my second title
            >>> print(rcParams['plotter.baseplotter.title'])
            my second title

        Furthermore, changing the :attr:`replace` attribute will change how you
        can access the keys::

            >>> d.replace = False

            # now setting d['title'] = 'anything' would raise an error (since
            # d.trace is set to True and 'title' is not a key in the rcParams
            # dictionary. Instead we need
            >>> d['plotter.baseplotter.title'] = 'anything'

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
        e = None
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
            except KeyError as e:
                pass
        if not found:
            if e is not None:
                raise
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
        depr = self._all_deprecated
        return dict((key, val[1]) for key, val in
                    six.iteritems(self.defaultParams)
                    if key not in depr)

    @property
    def descriptions(self):
        """The description of each keyword in the rcParams dictionary"""
        return {key: val[2] for key, val in six.iteritems(self.defaultParams)
                if len(val) >= 3}

    HEADER = """Configuration parameters of the psyplot module

You can copy this file (or parts of it) to another path and save it as
PSYPLOTRC. The directory should then be stored in the PSYPLOTCONFIGDIR
environment variable."""

    msg_depr = "%s is deprecated and replaced with %s; please use the latter."
    msg_depr_ignore = "%s is deprecated and ignored. Use %s"

    #: possible connections that shall be called if the rcParams value change
    _connections = defaultdict(list)

    @property
    def _all_deprecated(self):
        return set(chain(self._deprecated_ignore_map, self._deprecated_map))

    @property
    def defaultParams(self):
        return getattr(self, '_defaultParams', defaultParams)

    @defaultParams.setter
    def defaultParams(self, value):
        self._defaultParams = value

    @defaultParams.deleter
    def defaultParams(self):
        del self._defaultParams

    # validate values on the way in
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        defaultParams: dict
            The defaultParams to use (see the :attr:`defaultParams` attribute).
            By default, the :attr:`psyplot.config.rcsetup.defaultParams`
            dictionary is used

        Other Parameters
        ----------------
        ``*args, **kwargs``
            Any key-value pair for the initialization of the dictionary
        """
        defaultParams = kwargs.pop('defaultParams', None)
        if defaultParams is not None:
            self.defaultParams = defaultParams
        self._deprecated_map = {}
        self._deprecated_ignore_map = {}
        for k, v in six.iteritems(dict(*args, **kwargs)):
            try:
                self[k] = v
            except (ValueError, RuntimeError):
                # force the issue
                warn(_rcparam_warn_str.format(key=repr(k), value=repr(v),
                                              func='__init__'))
                dict.__setitem__(self, k, v)

    def __setitem__(self, key, val):
        key, val = self._get_depreceated(key, val)
        if key is None:
            return
        try:
            cval = self.validate[key](val)
        except ValueError as ve:
            raise ValueError("Key %s: %s" % (key, str(ve)))
        dict.__setitem__(self, key, cval)
        for func in self._connections.get(key, []):
            func(cval)

    def _get_depreceated(self, key, *args):
        if key in self._deprecated_map:
            alt_key, alt_val = self._deprecated_map[key]
            warn(self.msg_depr % (key, alt_key))
            key = alt_key
            return key, alt_val(args[0]) if args else None
        elif key in self._deprecated_ignore_map:
            alt = self._deprecated_ignore_map[key]
            warn(self.msg_depr_ignore % (key, alt))
            return None, None
        elif key not in self.defaultParams:
            raise KeyError(
                '%s is not a valid rc parameter. See rcParams.keys() for a '
                'list of valid parameters.' % (key,))
        return key, args[0] if args else None

    def __getitem__(self, key):
        key = self._get_depreceated(key)[0]
        if key is not None:
            return dict.__getitem__(self, key)

    def connect(self, key, func):
        key = self._get_depreceated(key)[0]
        if key is not None:
            self._connections[key].append(func)

    def remove(self, key, func):
        key = self._get_depreceated(key)[0]
        if key is not None:
            self._connections[key].remove(func)

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

    def update_from_defaultParams(self, defaultParams=None):
        """Update from the a dictionary like the :attr:`defaultParams`"""
        if defaultParams is None:
            defaultParams = self.defaultParams
        self.update({key: val[0] for key, val in defaultParams.items()})

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
        ret = RcParams()
        ret.defaultParams = self.defaultParams
        ret.update((key, value) for key, value in self.items()
                   if pattern_re.search(key))
        return ret

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
        The syntax is the same as for the initialization of the
        :class:`SubDict` class::

            >>> from psyplot import rcParams
            >>> d = rcParams.find_and_replace(['plotter.baseplotter.',
            ...                                'plotter.vector.'])
            >>> print(d['title'])
            None

            >>> print(d['arrowsize'])
            1.0

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
             exclude_keys=['project.plotters'], include_descriptions=True,
             **kwargs):
        """Dump this instance to a yaml file

        Parameters
        ----------
        fname: str or None
            file name to write to. If None, the string that would be written
            to a file is returned
        overwrite: bool
            If True and `fname` already exists, it will be overwritten
        include_keys: None or list of str
            Keys in the dictionary to be included. If None, all keys are
            included
        exclude_keys: list of str
            Keys from the :class:`RcParams` instance to be excluded

        Other Parameters
        ----------------
        ``**kwargs``
            Any other parameter for the :func:`yaml.dump` function

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
        if six.PY2:
            kwargs.setdefault('encoding', 'utf-8')
        d = {key: val for key, val in six.iteritems(self) if (
                include_keys is None or key in include_keys) and
             key not in exclude_keys}
        if include_descriptions:
            s = yaml.dump(d, **kwargs)
            desc = self.descriptions
            i = 2
            header = self.HEADER.splitlines() + [
                '', 'Created with python', ''] + sys.version.splitlines() + [
                    '', '']
            lines = ['# ' + l for l in header] + s.splitlines()
            for l in lines[2:]:
                key = l.split(':')[0]
                if key in desc:
                    lines.insert(i, '# ' + '\n# '.join(desc[key].splitlines()))
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

    def load_plugins(self, group='psyplot', raise_error=False):
        """
        Load the plotters and defaultParams from the plugins

        This method loads the `plotters` attribute and `defaultParams`
        attribute from the plugins that use the entry point specified by
        `group`. Entry points must be objects (or modules) that have a
        `defaultParams` and a `plotters` attribute.

        Parameters
        ----------
        group: str
            The group of the entry point
        raise_error: bool
            If True, an error is raised when multiple plugins define the same
            plotter or rcParams key. Otherwise only a warning is raised

        Returns
        -------
        dict
            The plotters configuration dictionaries from the plugins for the
            :func:`psyplot.project.register_plotter` function"""
        from pkg_resources import iter_entry_points
        import logging
        logger = logging.getLogger(__name__)
        plotters = self['project.plotters']
        def_plots = {}
        defaultParams = self.defaultParams
        def_keys = {'default': defaultParams}

        for ep in iter_entry_points(group=group, name='plugin'):
            logger.debug('Loading entrypoint %s', ep)
            plugin_mod = ep.load()
            rc = plugin_mod.rcParams

            # load the plotters
            plugin_plotters = rc.get('project.plotters', {})
            already_defined = set(plotters).intersection(plugin_plotters)
            if already_defined:
                msg = ("Error while loading psyplot plugin %s! The "
                       "following plotters have already been "
                       "defined:") % ep
                msg += '\n' + '\n'.join(chain.from_iterable(
                    (('%s by %s' % (key, plugin)
                      for plugin, keys in def_plots.items() if key in keys)
                     for key in already_defined)))
                if raise_error:
                    raise ImportError(msg)
                else:
                    warn(msg)
            plotters.update(plugin_plotters)

            # load the defaultParams keys
            plugin_defaultParams = rc.defaultParams
            already_defined = set(defaultParams).intersection(
                plugin_defaultParams) - {'project.plotters'}
            if already_defined:
                msg = ("Error while loading psyplot plugin %s! The "
                       "following default keys have already been "
                       "defined:") % ep
                msg += '\n' + '\n'.join(chain.from_iterable(
                    (('%s by %s' % (key, plugin)
                      for plugin, keys in def_keys.items() if key in keys)
                     for key in already_defined)))
                if raise_error:
                    raise ImportError(msg)
                else:
                    warn(msg)
            update_keys = set(plugin_defaultParams) - {'project.plotters'}
            def_keys[ep] = update_keys
            self.defaultParams.update(
                {key: plugin_defaultParams[key] for key in update_keys})

            # load the rcParams (without validation)
            super(RcParams, self).update({key: rc[key] for key in update_keys})

            # add the deprecated keys
            self._deprecated_ignore_map.update(rc._deprecated_ignore_map)
            self._deprecated_map.update(rc._deprecated_map)

    def copy(self):
        """Make sure, the right class is retained"""
        return RcParams(self)


def psyplot_fname(env_key='PSYPLOTRC', fname='psyplotrc.yml'):
    """
    Get the location of the config file.

    The file location is determined in the following order

    - `$PWD/psyplotrc.yml`

    - environment variable `PSYPLOTRC` (pointing to the file location or a
      directory containing the file `psyplotrc.yml`)

    - `$PSYPLOTCONFIGDIR/psyplot`

    - On Linux,

          - `$HOME/.config/psyplot/psyplotrc.yml`

    - On other platforms,

         - `$HOME/.psyplot/psyplotrc.yml` if `$HOME` is defined.

    - Lastly, it looks in `$PSYPLOTDATA/psyplotrc.yml` for a
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
    full_fname = os.path.join(cwd, fname)
    if os.path.exists(full_fname):
        return full_fname

    if env_key in os.environ:
        path = os.environ[env_key]
        if os.path.exists(path):
            if os.path.isdir(path):
                full_fname = os.path.join(path, fname)
                if os.path.exists(full_fname):
                    return full_fname
            else:
                return path

    configdir = get_configdir()
    if configdir is not None:
        full_fname = os.path.join(configdir, fname)
        if os.path.exists(full_fname):
            return full_fname

    return None


def get_configdir():
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

    if not os.path.exists(p):
        os.makedirs(p)
    return p


def validate_path_exists(s):
    """If s is a path, return s, else False"""
    if s is None:
        return None
    if os.path.exists(s):
        return s
    else:
        raise ValueError('"%s" should be a path but it does not exist' % s)


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
        d = validate_path_exists(d)
        try:
            with open(d) as f:
                return dict(yaml.load(f))
        except:
            raise ValueError("Could not convert {} to dictionary!".format(d))


def validate_bool_maybe_none(b):
    'Convert b to a boolean or raise'
    if isinstance(b, six.string_types):
        b = b.lower()
    if b is None or b == 'none':
        return None
    return validate_bool(b)


def validate_bool(b):
    """Convert b to a boolean or raise"""
    if isinstance(b, six.string_types):
        b = b.lower()
    if b in ('t', 'y', 'yes', 'on', 'true', '1', 1, True):
        return True
    elif b in ('f', 'n', 'no', 'off', 'false', '0', 0, False):
        return False
    else:
        raise ValueError('Could not convert "%s" to boolean' % b)


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
            return list(map(validate_str, s))
        except TypeError as e:
            raise ValueError(e.message)


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


#: :class:`dict` with default values and validation functions
defaultParams = {
    # user defined plotter keys
    'plotter.user': [
        {}, validate_dict,
        dedents("""
        formatoption keys and values that are defined by the user to be used by
        the specified plotters. For example to modify the title of all
        :class:`psyplot.plotter.maps.FieldPlotter` instances, set
        ``{'plotter.fieldplotter.title': 'my title'}``""")],

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

    # specify automatic drawing and showing of figures
    'auto_draw': [True, validate_bool,
                  ('Automatically draw the figures if the draw keyword in the '
                   'update and start_update methods is None')],
    'auto_show': [False, validate_bool,
                  ('Automatically show the figures after the update and'
                   'start_update methods')],

    # data
    'datapath': [None, validate_path_exists, 'path for supplementary data'],

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
    'project.import_seaborn': [
        None, validate_bool_maybe_none,
        'boolean controlling whether the seaborn module shall be imported '
        'when importing the project module. If None, it is only tried to '
        'import the module.'],
    'project.plotters': [
        {}, validate_dict,
        'mapping from identifier to plotter definitions for the Project class.'
        ' See the :func:`psyplot.project.register_plotter` function for '
        'possible keywords and values. See '
        ':attr:`psyplot.project.registered_plotters` for examples.'],
    }


_rcparam_warn_str = ("Trying to set {key} to {value} via the {func} "
                     "method of RcParams which does not validate cleanly. ")


_seq_err_msg = ('You must supply exactly {n:d} values, you provided '
                '{num:d} values: {s}')


_str_err_msg = ('You must supply exactly {n:d} comma-separated values, '
                'you provided '
                '{num:d} comma-separated values: {s}')

#: :class:`~psyplot.config.rcsetup.RcParams` instance that stores default
#: formatoptions and configuration settings.
rcParams = RcParams()
rcParams.update_from_defaultParams()
rcParams.load_from_file()
