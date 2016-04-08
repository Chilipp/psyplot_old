"""Sphinx extension that disables the displaying of large objects

By default the :class:`sphinx.ext.autodoc.DataDocumenter` and
:class:`sphinx.ext.autodoc.AttributeDocumenter` display the data in, e. g.,
dictionaries in the documentation. This extension gives the possibility to
choose which data shall be shown. It provides two additional configuration
values when included as an extension in the ``conf.py`` script:

- ``document_data``: Default: ``[]``. A list containing regex patterns of the
  data that shall be shown
- ``not_document_data``: Default: ``['.*']``. A list containing the regex
  patterns of the data that shall not be shown (by default, all)."""
import sphinx
from sphinx.ext.autodoc import Options, AutoDirective
try:
    from psyplot.sphinxext.extended_autodoc import (
        CallableDataDocumenter as DataDocumenter,
        CallableAttributeDocumenter as AttributeDocumenter)
    psyplot_documenters = True
except ImportError:
    from sphinx.ext.autodoc import DataDocumenter, AttributeDocumenter
    psyplot_documenters = False
import re

try:
    from psyplot.plotter import FormatoptionMeta
except ImportError:
    FormatoptionMeta = ()


def dont_document_obj(config, fullname):
    return (any(re.match(p, fullname) for p in config.not_document_data) and
            (not config.document_data or
             not any(re.match(p, fullname) for p in config.document_data)))


class NoDataDataDocumenter(DataDocumenter):
    """DataDocumenter that prevents the displaying of large data"""
    #: slightly higher priority as the one of the DataDocumenter
    priority = DataDocumenter.priority + 0.1

    def __init__(self, *args, **kwargs):
        super(NoDataDataDocumenter, self).__init__(*args, **kwargs)
        fullname = '.'.join(self.name.rsplit('::', 1))
        if hasattr(self.env, 'config') and dont_document_obj(
                self.env.config, fullname):
            self.options = Options(self.options)
            self.options.annotation = ' '


class NoDataAttributeDocumenter(AttributeDocumenter):
    """AttributeDocumenter that prevents the displaying of large data"""
    #: slightly higher priority as the one of the AttributeDocumenter
    priority = AttributeDocumenter.priority + 0.1

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        """Class method to determine whether the object can be documented

        We override the
        :meth:`sphinx.ext.autodoc.AttributeDocumenter.can_document_member`
        method to make sure that :class:`~psyplot.plotter.Formatoption` classes
        (created by the :class:`~psyplot.plotter.FormatoptionMeta` meta class
        are not regarded as attribute)"""
        if FormatoptionMeta and isinstance(member, FormatoptionMeta):
            # prevent Attribute Documenter from documenting Formatoption
            # classes
            return False
        else:
            return AttributeDocumenter.can_document_member(member, membername,
                                                           isattr, parent)

    def __init__(self, *args, **kwargs):
        super(NoDataAttributeDocumenter, self).__init__(*args, **kwargs)
        fullname = '.'.join(self.name.rsplit('::', 1))
        if hasattr(self.env, 'config') and dont_document_obj(
                self.env.config, fullname):
            self.options = Options(self.options)
            self.options.annotation = ' '


def setup(app):
    """setup function for using this module as a sphinx extension"""
    try:
        app.add_config_value('autodata_content', 'both', True)
    except sphinx.errors.ExtensionError:  # already registered
        pass
    # make sure to allow inheritance when registering new documenters
    for cls in [NoDataDataDocumenter, NoDataAttributeDocumenter]:
        if not issubclass(AutoDirective._registry.get(cls.objtype), cls):
            app.add_autodocumenter(cls)
    app.add_config_value('document_data', [], True)
    app.add_config_value('not_document_data', [re.compile('.*')], True)
    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}
