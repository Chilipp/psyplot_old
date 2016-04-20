"""Sphinx extension that defines a new automodule directive with autosummary

The :class:`AutoSummDirective` defined in this extension module allows the
same functionality as the automodule and autoclass directives of the
:mod:`sphinx.ext.autodoc` module but with an additional `autosummary` and a
`show-formatoptions` option. The first one puts a preceding autosummary
in the style of the :mod:`sphinx.ext.autosummary` module at the beginning of
the class or module, the latter allows a nice overview of the formatoptions
of a :class:`psyplot.plotter.Plotter`.
The content of this autosummary is automatically determined by the results of
the automodule (or autoclass) directive.

.. note::

    When used as a sphinx extension, this module overwrites the `automodule`
    directive of the :mod:`sphinx.ext.autodoc` module. Hence, if you want to
    use the :mod:`~sphinx.ext.autodoc` module and this module at the same time,
    this module should be listed after the autodoc module in the extensions
    list.
"""
import re
import six
import sphinx
from collections import defaultdict
from itertools import chain
from sphinx.ext.autodoc import (
    ClassDocumenter, ModuleDocumenter, ALL, AutoDirective, PycodeError,
    ModuleAnalyzer, bool_option, DataDocumenter, AttributeDocumenter,
    is_builtin_class_method, formatargspec, getargspec, force_decode,
    prepare_docstring)
import inspect
import sphinx.ext.autodoc as ad
from sphinx.ext.autosummary import Autosummary, ViewList, mangle_signature
from docutils import nodes
from psyplot.compat.pycompat import OrderedDict, map, filterfalse
try:
    from psyplot.plotter import Formatoption
except ImportError:
    pass

sphinx_version = list(map(float, re.findall('\d+', sphinx.__version__)[:3]))


class AutosummaryDocumenter(object):
    """Abstract class for for extending Documenter methods

    This classed is used as a base class for Documenters in order to provide
    the necessary methods for the :class:`AutoSummDirective`."""

    def __init__(self):
        raise NotImplementedError

    def get_grouped_documenters(self, all_members=False):
        """Method to return the member documenters

        This method is somewhat like a combination of the
        :meth:`sphinx.ext.autodoc.ModuleDocumenter.generate` method and the
        :meth:`sphinx.ext.autodoc.ModuleDocumenter.document_members` method.
        Hence it initializes this instance by importing the object, etc. and
        it finds the documenters to use for the autosummary option in the same
        style as the document_members does it.

        Returns
        -------
        dict
            dictionary whose keys are determined by the :attr:`member_sections`
            dictionary and whose values are lists of tuples. Each tuple
            consists of a documenter and a boolean to identify whether a module
            check should be made describes an attribute or not. The dictionary
            can be used in the
            :meth:`AutoSummDirective.get_items_from_documenters` method

        Notes
        -----
        If a :class:`sphinx.ext.autodoc.Documenter.member_order` value is not
        in the :attr:`member_sections` dictionary, it will be put into an
        additional `Miscallaneous` section."""
        self.parse_name()
        self.import_object()
        # If there is no real module defined, figure out which to use.
        # The real module is used in the module analyzer to look up the module
        # where the attribute documentation would actually be found in.
        # This is used for situations where you have a module that collects the
        # functions and classes of internal submodules.
        self.real_modname = None or self.get_real_modname()

        if not self.options.get('show-formatoptions') or not hasattr(
                self.object, '_get_formatoptions'):
            def is_not_fmt(docu_tuple):
                return True
        else:
            fmt_keys = list(self.object._get_formatoptions())

            def is_not_fmt(docu_tuple):
                return docu_tuple[0].name.split('.')[-1] not in fmt_keys

        # try to also get a source code analyzer for attribute docs
        try:
            self.analyzer = ModuleAnalyzer.for_module(self.real_modname)
            # parse right now, to get PycodeErrors on parsing (results will
            # be cached anyway)
            self.analyzer.find_attr_docs()
        except PycodeError as err:
            self.env.app.debug('[autodoc] module analyzer failed: %s', err)
            # no source file -- e.g. for builtin and C modules
            self.analyzer = None
            # at least add the module.__file__ as a dependency
            if hasattr(self.module, '__file__') and self.module.__file__:
                self.directive.filename_set.add(self.module.__file__)
        else:
            self.directive.filename_set.add(self.analyzer.srcname)

        self.env.temp_data['autodoc:module'] = self.modname
        if self.objpath:
            self.env.temp_data['autodoc:class'] = self.objpath[0]

        want_all = all_members or self.options.inherited_members or \
            self.options.members is ALL
        # find out which members are documentable
        members_check_module, members = self.get_object_members(want_all)

        # remove members given by exclude-members
        if self.options.exclude_members:
            members = [(membername, member) for (membername, member) in members
                       if membername not in self.options.exclude_members]

        # document non-skipped members
        memberdocumenters = []
        for (mname, member, isattr) in self.filter_members(members, want_all):
            classes = [cls for cls in six.itervalues(AutoDirective._registry)
                       if cls.can_document_member(member, mname, isattr, self)]
            if not classes:
                # don't know how to document this member
                continue
            # prefer the documenter with the highest priority
            classes.sort(key=lambda cls: cls.priority)
            # give explicitly separated module name, so that members
            # of inner classes can be documented
            full_mname = self.modname + '::' + \
                '.'.join(self.objpath + [mname])
            documenter = classes[-1](self.directive, full_mname, self.indent)
            memberdocumenters.append((documenter,
                                      members_check_module and not isattr))
        documenters = self.member_sections.copy()
        for section, order in six.iteritems(documenters):
            documenters[section] = sorted(
                (e for e in memberdocumenters
                 if e[0].member_order == order and is_not_fmt(e)),
                key=lambda e: e[0].member_order)
        fmts = defaultdict(set)
        for e in filterfalse(is_not_fmt, memberdocumenters):
            e[0].parse_name()
            e[0].import_object()
            fmts[e[0].object.groupname].add(e)
        for gname, l in sorted(fmts.items()):
            documenters[gname] = sorted(
                l, key=lambda e: e[0].object.key)
        remaining = sorted(
            (e for e in memberdocumenters
             if e[0].member_order not in six.itervalues(self.member_sections)),
            key=lambda e: e[0].name.split('::')[-1])
        if remaining:
            documenters['Miscallaneous'] = remaining
        return documenters


class AutoSummModuleDocumenter(ModuleDocumenter, AutosummaryDocumenter):
    """Module documentor suitable for the :class:`AutoSummDirective`

    This class has the same functionality as the base
    :class:`sphinx.ext.autodoc.ModuleDocumenter` class but with an additional
    `autosummary` and `show-formatoptions` option and the
    :meth:`get_grouped_documenters` method.
    It's priority is slightly higher than the one of the ModuleDocumenter

    Notes
    -----
    The `show-formatoptions` option has no effect for the automodule
    directive except that the `member` option is set."""

    #: slightly higher priority than
    #: :class:`sphinx.ext.autodoc.ModuleDocumenter`
    priority = ModuleDocumenter.priority + 0.1

    #: original option_spec from :class:`sphinx.ext.autodoc.ModuleDocumenter`
    #: but with additional autosummary boolean option
    option_spec = ModuleDocumenter.option_spec
    option_spec['autosummary'] = bool_option
    option_spec['show-formatoptions'] = bool_option  # does nothing

    member_sections = OrderedDict([
        ('Classes', ad.ClassDocumenter.member_order),
        ('Exceptions', ad.ExceptionDocumenter.member_order),
        ('Functions', ad.FunctionDocumenter.member_order),
        ('Data', ad.DataDocumenter.member_order),
        ])
    """:class:`~collections.OrderedDict` that includes the autosummary sections

    This dictionary defines the sections for the autosummmary option. The
    values correspond to the :attr:`sphinx.ext.autodoc.Documenter.member_order`
    attribute that shall be used for each section."""


class AutoSummClassDocumenter(ClassDocumenter, AutosummaryDocumenter):
    """Class documentor suitable for the :class:`AutoSummDirective`

    This class has the same functionality as the base
    :class:`sphinx.ext.autodoc.ClassDocumenter` class but with an additional
    `autosummary` and `show-formatoptions` option to provide the ability to
    provide a summary of all methods and attributes at the beginning and to
    show the formatoptions of a :class:`psyplot.plotters.Plotter` subclass.
    It's priority is slightly higher than the one of the ClassDocumenter"""

    #: slightly higher priority than
    #: :class:`sphinx.ext.autodoc.ClassDocumenter`
    priority = ClassDocumenter.priority + 0.1

    #: original option_spec from :class:`sphinx.ext.autodoc.ClassDocumenter`
    #: but with additional autosummary and show-formatoption boolean option
    option_spec = ClassDocumenter.option_spec
    option_spec['autosummary'] = bool_option
    option_spec['show-formatoptions'] = bool_option

    member_sections = OrderedDict([
        ('Methods', ad.MethodDocumenter.member_order),
        ('Attributes', ad.AttributeDocumenter.member_order),
        ])
    """:class:`~collections.OrderedDict` that includes the autosummary sections

    This dictionary defines the sections for the autosummmary option. The
    values correspond to the :attr:`sphinx.ext.autodoc.Documenter.member_order`
    attribute that shall be used for each section."""

    def filter_members(self, *args, **kwargs):
        ret = super(AutoSummClassDocumenter, self).filter_members(
            *args, **kwargs)
        if self.options.get('show-formatoptions') and hasattr(
                self.object, '_get_formatoptions'):
            fmt_members = defaultdict(set)
            all_fmt = set(self.object._get_formatoptions())
            for i, (mname, member, isattr) in enumerate(ret):
                if isinstance(member, Formatoption):
                    fmt_members[member.group].add((mname, member, isattr))
                    all_fmt.remove(mname)
            for fmt in all_fmt:
                fmto = getattr(self.object, fmt)
                fmt_members[fmto.group].add((fmt, fmto, True))
            ret.extend(
                (tup for tup in chain(*map(sorted, fmt_members.values()))
                 if tup not in ret))
        return ret


class CallableDataDocumenter(DataDocumenter):
    """:class:`sphinx.ext.autodoc.DataDocumenter` that uses the __call__ attr
    """

    priority = DataDocumenter.priority + 0.1

    def format_args(self):
        # for classes, the relevant signature is the __init__ method's
        callmeth = self.get_attr(self.object, '__call__', None)
        if callmeth is None:
            return None
        try:
            argspec = getargspec(callmeth)
        except TypeError:
            # still not possible: happens e.g. for old-style classes
            # with __call__ in C
            return None
        if argspec[0] and argspec[0][0] in ('cls', 'self'):
            del argspec[0][0]
        if sphinx_version < [1, 4]:
            return formatargspec(*argspec)
        else:
            return formatargspec(callmeth, *argspec)

    def get_doc(self, encoding=None, ignore=1):
        """Reimplemented  to include data from the call method"""
        content = self.env.config.autodata_content
        if content not in ('both', 'call') or not self.get_attr(
                self.get_attr(self.object, '__call__', None), '__doc__'):
            return super(CallableDataDocumenter, self).get_doc(
                encoding=encoding, ignore=ignore)

        # for classes, what the "docstring" is can be controlled via a
        # config value; the default is both docstrings
        docstrings = []
        if content != 'call':
            docstring = self.get_attr(self.object, '__doc__', None)
            docstrings = [docstring] if docstring else []
        calldocstring = self.get_attr(
            self.get_attr(self.object, '__call__', None), '__doc__')
        docstrings.append(calldocstring)
        doc = []
        for docstring in docstrings:
            if not isinstance(docstring, six.text_type):
                docstring = force_decode(docstring, encoding)
            doc.append(prepare_docstring(docstring))

        return doc


class CallableAttributeDocumenter(AttributeDocumenter):
    """:class:`sphinx.ext.autodoc.DataDocumenter` that uses the __call__ attr
    """

    priority = AttributeDocumenter.priority + 0.1

    def format_args(self):
        # for classes, the relevant signature is the __init__ method's
        callmeth = self.get_attr(self.object, '__call__', None)
        if callmeth is None:
            return None
        try:
            argspec = getargspec(callmeth)
        except TypeError:
            # still not possible: happens e.g. for old-style classes
            # with __call__ in C
            return None
        if argspec[0] and argspec[0][0] in ('cls', 'self'):
            del argspec[0][0]
        if sphinx_version < [1, 4]:
            return formatargspec(*argspec)
        else:
            return formatargspec(callmeth, *argspec)

    def get_doc(self, encoding=None, ignore=1):
        """Reimplemented  to include data from the call method"""
        content = self.env.config.autodata_content
        if content not in ('both', 'call') or not self.get_attr(
                self.get_attr(self.object, '__call__', None), '__doc__'):
            return super(CallableAttributeDocumenter, self).get_doc(
                encoding=encoding, ignore=ignore)

        # for classes, what the "docstring" is can be controlled via a
        # config value; the default is both docstrings
        docstrings = []
        if content != 'call':
            docstring = self.get_attr(self.object, '__doc__', None)
            docstrings = [docstring + '\n'] if docstring else []
        calldocstring = self.get_attr(
            self.get_attr(self.object, '__call__', None), '__doc__')
        if docstrings:
            docstrings[0] += calldocstring
        else:
            docstrings.append(calldocstring + '\n')

        doc = []
        for docstring in docstrings:
            if not isinstance(docstring, six.text_type):
                docstring = force_decode(docstring, encoding)
            doc.append(prepare_docstring(docstring, ignore))

        return doc


class AutoSummDirective(AutoDirective, Autosummary):
    """automodule directive that makes a summary at the beginning of the module

    This directive combines the :class:`sphinx.ext.autodoc.AutoDirective` and
    :class:`sphinx.ext.autosummary.Autosummary` directives to put a summary of
    the specified module at the beginning of the module documentation."""

    _default_flags = AutoDirective._default_flags.union({'autosummary',
                                                         'show-formatoptions'})

    @property
    def autosummary_documenter(self):
        """Returns the AutosummaryDocumenter subclass that can be used"""
        try:
            return self._autosummary_documenter
        except:
            pass
        objtype = self.name[4:]
        doc_class = self._registry[objtype]
        documenter = doc_class(self, self.arguments[0])
        if hasattr(documenter, 'get_grouped_documenters'):
            self._autosummary_documenter = documenter
            return documenter
        if objtype == 'module':
            documenter = AutoSummModuleDocumenter(self, self.arguments[0])
        elif objtype == 'class':
            documenter = AutoSummClassDocumenter(self, self.arguments[0])
        else:
            raise ValueError(
                "Could not find a valid documenter for the object type %s" % (
                    objtype))
        self._autosummary_documenter = documenter
        return documenter

    def run(self):
        """Run method for the directive"""
        doc_nodes = AutoDirective.run(self)
        if 'autosummary' not in self.options:
            return doc_nodes
        self.warnings = []
        self.env = self.state.document.settings.env
        self.result = ViewList()
        documenter = self.autosummary_documenter
        grouped_documenters = documenter.get_grouped_documenters()
        if 'show-formatoptions' in self.env.config.autodoc_default_flags:
            self.options['show-formatoptions'] = True
        summ_nodes = self.autosumm_nodes(documenter, grouped_documenters)

        dn = summ_nodes.pop(documenter.fullname)
        doc_nodes = self.inject_summ_nodes(doc_nodes, summ_nodes)
        # insert the nodes directly after the paragraphs
        if self.name == 'autoclass':
            for node in dn[::-1]:
                self._insert_after_paragraphs(doc_nodes[1], node)
            dn = []
        elif self.name == 'automodule':
            # insert table before the documentation of the members
            istart = 2 if 'noindex' not in self.options else 0
            found = False
            if len(doc_nodes[istart:]) >= 2:
                for i in range(istart, len(doc_nodes)):
                    if isinstance(doc_nodes[i], sphinx.addnodes.index):
                        found = True
                        break
            if found:
                for node in dn[::-1]:
                    doc_nodes.insert(i, node)
                dn = []
        return self.warnings + dn + doc_nodes

    def _insert_after_paragraphs(self, node, insertion):
        """Inserts the given `insertion` node after the paragraphs in `node`

        This method inserts the `insertion` node after the instances of
        nodes.paragraph in the given `node`.
        Usually the node of one documented class is set up like

        Name of the documented item (allways) (nodes.Element)
        Summary (sometimes) (nodes.paragraph)
        description (sometimes) (nodes.paragraph)
        Parameters section (sometimes) (nodes.rubric)

        We want to be below the description, so we loop until we
        are below all the paragraphs. IF that does not work,
        we simply put it at the end"""
        found = False
        if len(node) >= 2:
            for i in range(len(node[1])):
                if not isinstance(node[1][i], nodes.paragraph):
                    node[1].insert(i + 1, insertion)
                    found = True
                    break
        if not found:
            node.insert(1, insertion)

    def inject_summ_nodes(self, doc_nodes, summ_nodes):
        """Method to inject the autosummary nodes into the autodoc nodes

        Parameters
        ----------
        doc_nodes: list
            The list of nodes as they are generated by the
            :meth:`sphinx.ext.autodoc.AutoDirective.run` method
        summ_nodes: dict
            The generated autosummary nodes as they are generated by the
            :meth:`autosumm_nodes` method. Note that `summ_nodes` must only
            contain the members autosummary tables!

        Returns
        -------
        doc_nodes: list
            The modified `doc_nodes`

        Notes
        -----
        `doc_nodes` are modified in place and not copied!"""
        for i, node in enumerate(doc_nodes):
            # check if the node has a autosummary table in the summ_nodes
            if (len(node) and isinstance(node[0], nodes.Element) and
                    node[0].get('module') and node[0].get('fullname')):
                node_summ_nodes = summ_nodes.get("%s.%s" % (
                    node[0]['module'], node[0]['fullname']))
                if not node_summ_nodes:
                    continue
                for summ_node in node_summ_nodes[::-1]:
                    self._insert_after_paragraphs(node, summ_node)
        return doc_nodes

    def autosumm_nodes(self, documenter, grouped_documenters):
        """Create the autosummary nodes based on the documenter content

        Parameters
        ----------
        documenter: sphinx.ext.autodoc.Documenter
            The base (module or class) documenter for which to generate the
            autosummary tables of its members
        grouped_documenters: dict
            The dictionary as it is returned from the
            :meth:`AutosummaryDocumenter.get_grouped_documenters` method

        Returns
        -------
        dict
            a mapping from the objects fullname to the corresponding
            autosummary tables of its members. The objects include the main
            object of the given `documenter` and the classes that are defined
            in it

        See Also
        --------
        AutosummaryDocumenter.get_grouped_documenters, inject_summ_nodes"""

        summ_nodes = {}
        this_nodes = []
        for section, documenters in six.iteritems(grouped_documenters):
            items = self.get_items_from_documenters(documenters)
            if not items:
                continue
            node = nodes.rubric()
            # create note for the section title (we could also use .. rubric
            # but that causes problems for latex documentations)
            self.state.nested_parse(
                ViewList(['**%s**' % section]), 0, node)
            this_nodes += node
            this_nodes += self.get_table(items)
            for mdocumenter, check_module in documenters:
                if (mdocumenter.objtype == 'class' and
                        not (check_module and not mdocumenter.check_module())):
                    if hasattr(mdocumenter, 'get_grouped_documenters'):
                        summ_nodes.update(self.autosumm_nodes(
                            mdocumenter, mdocumenter.get_grouped_documenters())
                            )
        summ_nodes[documenter.fullname] = this_nodes
        return summ_nodes

    def get_items_from_documenters(self, documenters):
        """Return the items needed for creating the tables

        This method creates the items that are used by the
        :meth:`sphinx.ext.autosummary.Autosummary.get_table` method by what is
        taken from the values of the
        :meth:`AutoSummModuleDocumenter.get_grouped_documenters` method.

        Returns
        -------
        list
            A list containing tuples like
            ``(name, signature, summary_string, real_name)`` that can be used
            for the :meth:`sphinx.ext.autosummary.Autosummary.get_table`
            method."""

        items = []

        max_item_chars = 50
        base_documenter = self.autosummary_documenter
        base_documenter.analyzer = ModuleAnalyzer.for_module(
                base_documenter.real_modname)
        attr_docs = base_documenter.analyzer.find_attr_docs()

        for documenter, check_module in documenters:
            documenter.parse_name()
            documenter.import_object()
            documenter.real_modname = documenter.get_real_modname()
            real_name = documenter.fullname
            display_name = documenter.object_name
            if check_module and not documenter.check_module():
                continue

            # -- Grab the signature

            sig = documenter.format_signature()
            if not sig:
                sig = ''
            else:
                max_chars = max(10, max_item_chars - len(display_name))
                sig = mangle_signature(sig, max_chars=max_chars)
                sig = sig.replace('*', r'\*')

            # -- Grab the documentation

            no_docstring = False
            if documenter.objpath:
                key = ('.'.join(documenter.objpath[:-1]),
                       documenter.objpath[-1])
                try:
                    doc = attr_docs[key]
                    no_docstring = True
                except KeyError:
                    pass
            if not no_docstring:
                documenter.add_content(None)
                doc = documenter.get_doc()
                if doc:
                    doc = doc[0]
                else:
                    continue

            while doc and not doc[0].strip():
                doc.pop(0)

            # If there's a blank line, then we can assume the first sentence /
            # paragraph has ended, so anything after shouldn't be part of the
            # summary
            for i, piece in enumerate(doc):
                if not piece.strip():
                    doc = doc[:i]
                    break

            # Try to find the "first sentence", which may span multiple lines
            m = re.search(r"^([A-Z].*?\.)(?:\s|$)", " ".join(doc).strip())
            if m:
                summary = m.group(1).strip()
            elif doc:
                summary = doc[0].strip()
            else:
                summary = ''

            items.append((display_name, sig, summary, real_name))
        return items


def setup(app):
    """setup function for using this module as a sphinx extension"""
    try:
        app.add_config_value('autodata_content', 'both', True)
    except sphinx.errors.ExtensionError:  # value already registered
        pass
    # make sure to allow inheritance when registering new documenters
    for cls in [AutoSummClassDocumenter, AutoSummModuleDocumenter,
                CallableDataDocumenter, CallableAttributeDocumenter]:
        if not issubclass(AutoDirective._registry.get(cls.objtype), cls):
            app.add_autodocumenter(cls)
    app.add_directive('automodule', AutoSummDirective)
    app.add_directive('autoclass', AutoSummDirective)
    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}
