import types
import six
from matplotlib.docstring import dedent
from matplotlib.cbook import dedent as dedents
from re import (compile as re_compile, sub, MULTILINE, findall, finditer,
                VERBOSE)
from .warning import warn


def safe_modulo(s, meta, checked='', print_warning=True):
    """Safe version of the modulo operation (%) of strings

    Parameters
    ----------
    s: str
        string to apply the modulo operation with
    meta: dict
        meta informations to insert
    checked: {'KEY', 'VALUE'}, optional
        Security parameter for the recursive structure of this function. It can
        be set to 'VALUE' if an error shall be raised when facing a TypeError
        or ValueError or to 'KEY' if an error shall be raised when facing a
        KeyError
    print_warning: bool
        If True and a key is not existent in `s`, a warning is raised


    Examples
    --------
    .. ipython::
        :okexcept:
        :okwarning:

        In [1]: from psyplot.docstring import safe_modulo

        In [2]: s = "That's %(one)s string %(with)s missing 'with' and %s key"

        In [3]: s % {'one': 1}
        # raises KeyError because of missing 'with'

        In [4]: s% {'one': 1, 'with': 2}
        # raises TypeError because of '%s'

        In [5]: safe_modulo(s, {'one': 1})
        Out [5]: "That's 1 string %%(with)s missing 'with'"""
    try:
        return s % meta
    except (ValueError, TypeError, KeyError):
        # replace the missing fields by %%
        keys = finditer(r"""(?<!%)(%%)*%(?!%)   # uneven number of %
                            \((?P<key>(?s).*?)\)# key enclosed in brackets""",
                        s, VERBOSE)
        for m in keys:
            key = m.group('key')
            if not isinstance(meta, dict) or key not in meta:
                if print_warning:
                    warn("%r is not a valid key!" % key)
                full = m.group()
                s = s.replace(full, '%' + full)
        if checked != 'KEY':
            return safe_modulo(s, meta, checked='KEY',
                               print_warning=print_warning)
        if not isinstance(meta, dict) or checked == 'VALUE':
            raise
        s = sub(r"""(?<!%)(%%)*%(?!%) # uneven number of %
                    \s*(\w|$)         # format strings""", '%\g<0>', s,
                flags=VERBOSE)
        return safe_modulo(s, meta, checked='VALUE',
                           print_warning=print_warning)

dedentf = dedent


class DocStringProcessor(object):
    """Class that is intended to process docstrings

    It is, but only to minor extends, inspired by the
    :class:`matplotlib.docstring.Substitution` class.

    Examples
    --------
    .. ipython::

        In [1]: from psyplot.docstring import DocStringProcessor

        In [2]: d = DocStringProcessor(doc_key='My doc string')

        In [3]: @d
           ...: def doc_test():
           ...:     '''That's %(doc_key)s'''
           ...:     pass
           ...:

        In [4]: help(doc_test)
        Out [4]: That's my doc string"""

    #: :class:`dict`. Dictionary containing the compiled patterns to identify
    #: the Parameters, Other Parameters, Warnings and Notes sections in a
    #: docstring
    patterns = {}

    #: :class:`dict`. Dictionary containing the parameters that are used in for
    #: substitution.
    params = {}

    #: sections that behave the same as the `Parameter` section by defining a
    #: list
    param_like_sections = ['Parameters', 'Other Parameters', 'Returns',
                           'Possible types', 'Raises']
    #: sections that include (possibly unintended) text
    text_sections = ['Warnings', 'Notes', 'Examples', 'See Also',
                     'References']

    def __init__(self, *args, **kwargs):
        """
    Parameters
    ----------
    ``*args`` and ``**kwargs``
        Parameters that shall be used for the substitution. Note that you can
        only provide either ``*args`` or ``**kwargs``, furthermore most of the
        methods like `get_sectionsf` require ``**kwargs`` to be provided."""
        if len(args) and len(kwargs):
            raise ValueError("Only positional or keyword args are allowed")
        self.params = args or kwargs
        patterns = {}
        all_sections = self.param_like_sections + self.text_sections
        for section in self.param_like_sections:
            patterns[section] = re_compile(
                '(?<=%s\n%s\n)(?s)(.+?)(?=\n\n\S+|$)' % (
                    section, '-'*len(section)))
        all_sections_patt = '|'.join(
            '%s\n%s\n' % (s, '-'*len(s)) for s in all_sections)
        # examples and see also
        for section in self.text_sections:
            patterns[section] = re_compile(
                '(?<=%s\n%s\n)(?s)(.+?)(?=%s|$)' % (
                    section, '-'*len(section), all_sections_patt))
        self.patterns = patterns

    def __call__(self, func):
        func.__doc__ = func.__doc__ and safe_modulo(func.__doc__, self.params)
        return func

    @dedentf
    def get_sections(self, s, base, sections=['Parameters', 'Possible types',
                                              'Other Parameters']):
        """
        Method that extracts the specified sections out of the given string

        Parameters
        ----------
        s: str
            Docstring to split
        base: str
            base to use in the :attr:`sections` attribute
        sections: list of str
            sections to look for. Each section must be followed by a newline
            character ('\\n') and a bar of '-' (following the numpy (napoleon)
            docstring conventions).
        """
        params = self.params
        patterns = self.patterns
        for section in sections:
            key = '%s.%s' % (base, section.lower().replace(' ', '_'))
            try:
                params[key] = patterns[section].search(s).group(0).rstrip()
            except AttributeError:
                params[key] = ''
        return s

    @dedentf
    def get_sectionsf(self, *args, **kwargs):
        """
        Decorator method to extract sections from a function docstring

        ``*args`` and ``**kwargs`` are specified by the :meth:`get_sections`
        method. Note, that the first argument will be the docstring of the
        specified function

        Returns
        -------
        function
            Wrapper that takes a function as input and registers its sections
            via the :meth:`get_sections` method"""
        def func(f):
            doc = f.__doc__
            self.get_sections(doc or '', *args, **kwargs)
            return f
        return func

    @dedentf
    def dedent(self, func):
        """
        A special case of the DocStringProcessor first performs a dedent on the
        incoming docstring

        Parameters
        ----------
        func: function
            function with the documentation to dedent and whose sections
            shall be inserted from the :attr:`params` attribute"""
        if isinstance(func, types.MethodType) and not six.PY3:
            func = func.im_func
        return self(dedent(func))

    @dedentf
    def dedents(self, s):
        """
        A special case of the DocStringProcessor that first performs a dedent
        on the incoming string

        Parameters
        ----------
        s: str
            string to dedent and insert the sections of the :attr:`params`
            attribute"""
        s = dedents(s)
        return safe_modulo(s, self.params)

    @dedentf
    def delete_params(self, base_key, *params):
        """
        Method to delete a parameter from a parameter documentation.

        This method deletes the given `param` from the `base_key` item in the
        :attr:`params` dictionary and creates a new item with the original
        documentation without the description of the param. This method works
        for the ``'Parameters'`` sections.

        Parameters
        ----------
        base_key: str
            key in the :attr:`params` dictionary
        ``*params``
            str. Parameter identifier of which the documentations shall be
            deleted

        See Also
        --------
        delete_types"""
        patt = '|'.join(s + ':(?s).+?\n(?=\S+|$)' for s in params)
        self.params[base_key + '.no_' + '|'.join(params)] = sub(
            patt, '', self.params[base_key] + '\n', MULTILINE).rstrip()

    def delete_kwargs(self, base_key, args=None, kwargs=None):
        """
        Deletes the ``*args`` or ``**kwargs`` part from the parameters section

        Either `args` or `kwargs` must not be None. The resulting key will be
        stored in ``base_key + 'no_args_kwargs'

        Parameters
        ----------
        base_key: str
            The key in the :attr:`params` attribute to use
        args: None or str
            The string for the args to delete
        kwargs: None or str
            The string for the kwargs to delete"""
        if not args and not kwargs:
            warn("Neither args nor kwargs are given. I do nothing for %s" % (
                base_key))
            return
        types = []
        if args is not None:
            types.append('`?`?\*%s`?`?' % args)
        if kwargs is not None:
            types.append('`?`?\*\*%s`?`?' % kwargs)
        self.delete_types(base_key, 'no_args_kwargs', *types)

    @dedentf
    def delete_types(self, base_key, out_key, *types):
        """
        Method to delete a parameter from a parameter documentation.

        This method deletes the given `param` from the `base_key` item in the
        :attr:`params` dictionary and creates a new item with the original
        documentation without the description of the param. This method works
        for the ``'Possible Types'`` and ``'Results'`` sections.

        Parameters
        ----------
        base_key: str
            key in the :attr:`params` dictionary
        out_key: str
            Extension for the base key (the final key will be like
            ``'%s.%s' % (base_key, out_key)``
        ``*types``
            str. The type identifier of which the documentations shall deleted

        See Also
        --------
        delete_params"""
        patt = '|'.join(s + '\n(?s).+?\n(?=\S+|$)' for s in types)
        self.params['%s.%s' % (base_key, out_key)] = sub(
            patt, '', self.params[base_key] + '\n', MULTILINE).rstrip()

    @dedentf
    def keep_params(self, base_key, *params):
        """
        Method to keep only specific parameters from a parameter documentation.

        This method extracts the given `param` from the `base_key` item in the
        :attr:`params` dictionary and creates a new item with the original
        documentation with only the description of the param. This method works
        for the ``'Parameters'`` sections.

        Parameters
        ----------
        base_key: str
            key in the :attr:`params` dictionary
        ``*params``
            str. Parameter identifier of which the documentations shall be
            in the new section

        See Also
        --------
        keep_types"""
        patt = '|'.join(s + ':(?s).+?\n(?=\S+|$)' for s in params)
        self.params[base_key + '.' + '|'.join(params)] = ''.join(findall(
            patt, self.params[base_key] + '\n', MULTILINE)).rstrip()

    @dedentf
    def keep_types(self, base_key, out_key, *types):
        """
        Method to keep only specific parameters from a parameter documentation.

        This method extracts the given `type` from the `base_key` item in the
        :attr:`params` dictionary and creates a new item with the original
        documentation with only the description of the type. This method works
        for the ``'Possible Types'`` and ``'Results'`` sections.

        Parameters
        ----------
        base_key: str
            key in the :attr:`params` dictionary
        out_key: str
            Extension for the base key (the final key will be like
            ``'%s.%s' % (base_key, out_key)``
        ``*types``
            str. The type identifier of which the documentations shall be
            in the new section

        See Also
        --------
        keep_params"""
        patt = '|'.join(s + '\n(?s).+?\n(?=\S+|$)' for s in types)
        self.params['%s.%s' % (base_key, out_key)] = ''.join(findall(
            patt, self.params[base_key] + '\n', MULTILINE)).rstrip()

    @dedentf
    def save_docstring(self, key):
        """
        Descriptor method to save a docstring from a function

        Like the :meth:`get_sectionsf` method this method serves as a
        descriptor for functions but saves the whole docstring"""
        def func(f):
            self.params[key] = f.__doc__ or ''
            return f
        return func


#: :class:`DocStringProcessor` that simplifies the reuse of docstrings from
#: between different python objects.
docstrings = DocStringProcessor()
