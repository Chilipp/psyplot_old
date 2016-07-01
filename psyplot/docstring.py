import types
import six
from matplotlib.docstring import dedent
from matplotlib.cbook import dedent as dedents
from re import (compile as re_compile, sub, MULTILINE, findall,
                VERBOSE)
from psyplot.warning import warn

substitution_pattern = re_compile(
    r"""(?<!%)(%%)*%(?!%)   # uneven number of %
        \((?P<key>(?s).*?)\)# key enclosed in brackets""", VERBOSE)


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
    The effects are demonstrated by this example::

        >>> from psyplot.docstring import safe_modulo
        >>> s = "That's %(one)s string %(with)s missing 'with' and %s key"
        >>> s % {'one': 1}
        # raises KeyError because of missing 'with'
        >>> s% {'one': 1, 'with': 2}
        # raises TypeError because of '%s'
        >>> safe_modulo(s, {'one': 1})
        "That's 1 string %(with)s missing 'with' and %s key"
    """
    try:
        return s % meta
    except (ValueError, TypeError, KeyError):
        # replace the missing fields by %%
        keys = substitution_pattern.finditer(s)
        for m in keys:
            key = m.group('key')
            if not isinstance(meta, dict) or key not in meta:
                if print_warning:
                    warn("%r is not a valid key!" % key)
                full = m.group()
                s = s.replace(full, '%' + full)
        if 'KEY' not in checked:
            return safe_modulo(s, meta, checked=checked + 'KEY',
                               print_warning=print_warning)
        if not isinstance(meta, dict) or 'VALUE' in checked:
            raise
        s = sub(r"""(?<!%)(%%)*%(?!%) # uneven number of %
                    \s*(\w|$)         # format strings""", '%\g<0>', s,
                flags=VERBOSE)
        return safe_modulo(s, meta, checked=checked + 'VALUE',
                           print_warning=print_warning)

dedentf = dedent


class DocStringProcessor(object):
    """Class that is intended to process docstrings

    It is, but only to minor extends, inspired by the
    :class:`matplotlib.docstring.Substitution` class.

    Examples
    --------
    Create docstring processor via::

        >>> from psyplot.docstring import DocStringProcessor
        >>> d = DocStringProcessor(doc_key='My doc string')

    And then use it as a decorator to process the docstring::

        >>> @d
        ... def doc_test():
        ...     '''That's %(doc_key)s'''
        ...     pass

        >>> print(doc_test.__doc__)
        That's my doc string

    Use the :meth:`get_sectionsf` method to extract Parameter sections (or
    others) form the docstring for later usage (and make sure, that the
    docstring is dedented)::

        >>> @d.get_sectionsf('docstring_example')
        ... @d.dedent
        ... def doc_test(a=1, b=2):
        ...     '''
        ...     That's %(doc_key)s
        ...
        ...     Parameters
        ...     ----------
        ...     a: int, optional
        ...         A dummy parameter description
        ...     b: int, optional
        ...         A second dummy parameter
        ...
        ...     Examples
        ...     --------
        ...     Some dummy example doc'''
        ...     print(a)

        >>> @docstrings.dedent
        ... def second_test(a=1):
        ...     '''
        ...     My second function where I want to use the docstring from
        ...     above
        ...
        ...     Parameters
        ...     ----------
        ...     %(docstring_example.parameters)s
        ...
        ...     Examples
        ...     --------
        ...     %(docstring_example.examples)s'''
        ...     pass

        >>> print(second_test.__doc__)
        My second function where I want to use the docstring from
        above

        Parameters
        ----------
        a: int, optional
            A dummy parameter description
        b: int, optional
            A second dummy parameter

        Examples
        --------
        Some dummy example doc
    """

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
        Method that extracts the specified sections out of the given string if
        (and only if) the docstring follows the numpy documentation guidelines
        [1]_. Note that the section either must appear in the
        :attr:`param_like_sections` or the :attr:`text_sections` attribute.

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

        References
        ----------
        .. [1] https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

        See Also
        --------
        delete_params, keep_params, delete_types, keep_types, delete_kwargs:
            For manipulating the docstring sections
        save_docstring:
            for saving an entire docstring
        """
        params = self.params
        for section in sections:
            key = '%s.%s' % (base, section.lower().replace(' ', '_'))
            params[key] = self._get_section(s, section)
        return s

    def _get_section(self, s, section):
        try:
            return self.patterns[section].search(s).group(0).rstrip()
        except AttributeError:
            return ''

    @dedentf
    def get_sectionsf(self, *args, **kwargs):
        """
        Decorator method to extract sections from a function docstring

        Parameters
        ----------
        ``*args`` and ``**kwargs``
            See the :meth:`get_sections` method. Note, that the first argument
            will be the docstring of the specified function

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
        self.params[base_key + '.' + '|'.join(params)] = self._keep_params(
            self.params[base_key], params)

    def _keep_params(self, s, params):
        patt = '|'.join(s + ':(?s).+?\n(?=\S+|$)' for s in params)
        return ''.join(findall(patt, s.rstrip() + '\n', MULTILINE)).rstrip()

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
        self.params['%s.%s' % (base_key, out_key)] = self._keep_types(
            self.params[base_key], types)

    def _keep_types(self, s, types):
        patt = '|'.join(s + '\n(?s).+?\n(?=\S+|$)' for s in types)
        return ''.join(findall(patt, s.rstrip() + '\n', MULTILINE)).rstrip()

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

    @dedentf
    def get_summary(self, s, base=None):
        """
        Get the summary of the given docstring s

        This method extracts the summary from the given docstring `s` which is
        basicly the part until two newlines appear

        Parameters
        ----------
        s: str
            The docstring to use
        base: str or None
            A key under which the summary shall be stored in the :attr:`params`
            attribute. If not None, the summary will be stored in
            ``base + '.summary'``

        Returns
        -------
        str
            The extracted summary"""
        lines = s.splitlines()
        try:
            end = lines.index('')
            sl = slice(end)
        except ValueError:  # only summary in s
            sl = slice(None)
        summary = ' '.join(lines[sl])
        if base is not None:
            self.params[base + '.summary'] = summary
        return summary

    @dedentf
    def get_summaryf(self, *args, **kwargs):
        """
        Decorator method to extract summary from a function docstring

        Parameters
        ----------
        ``*args`` and ``**kwargs``
            See the :meth:`get_summary` method. Note, that the first argument
            will be the docstring of the specified function

        Returns
        -------
        function
            Wrapper that takes a function as input and registers its sections
            via the :meth:`get_sections` method"""
        def func(f):
            doc = f.__doc__
            self.get_summary(doc or '', *args, **kwargs)
            return f
        return func


def indent(text, num=4):
    """Indet the given string"""
    str_indent = ' ' * num
    return str_indent + ('\n' + str_indent).join(text.splitlines())


def append_original_doc(parent, num=0):
    """Return an iterator that append the docstring of the given `parent`
    function to the applied function"""
    def func(func):
        func.__doc__ = func.__doc__ and func.__doc__ + indent(
            parent.__doc__, num)
        return func
    return func


#: :class:`DocStringProcessor` instance that simplifies the reuse of docstrings
#: from between different python objects.
docstrings = DocStringProcessor()
