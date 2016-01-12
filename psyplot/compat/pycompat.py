"""Compatibility module for different python versions"""
import os
import six

if six.PY3:

    class DictMethods(object):

        @staticmethod
        def iteritems(d):
            return iter(dict.items(d))

        @staticmethod
        def itervalues(d):
            return iter(dict.values(d))

        @staticmethod
        def iterkeys(d):
            return iter(dict.keys(d))

    def getcwd(*args, **kwargs):
        return os.getcwd(*args, **kwargs)

    basestring = str
    unicode_type = str
    bytes_type = bytes
    range = range
    zip = zip
    filter = filter
    map = map
    from functools import reduce
    from itertools import filterfalse
    import builtins
elif six.PY2:
    # Python 2

    class DictMethods(object):
        @staticmethod
        def iteritems(d):
            return dict.iteritems(d)

        @staticmethod
        def itervalues(d):
            return dict.itervalues(d)

        @staticmethod
        def iterkeys(d):
            return dict.iterkeys(d)

    def getcwd(*args, **kwargs):
        return os.getcwdu(*args, **kwargs)

    basestring = basestring
    unicode_type = unicode
    bytes_type = str
    range = xrange
    from itertools import (izip as zip, imap as map, ifilter as filter,
                           ifilterfalse as filterfalse)
    reduce = reduce
    import __builtin__ as builtins

try:
    from cyordereddict import OrderedDict
except ImportError:
    try:
        from collections import OrderedDict
    except ImportError:
        from ordereddict import OrderedDict

try:
    from collections import UserDict
except ImportError:
    from UserDict import IterableUserDict as UserDict

def isstring(s):
    return isinstance(s, six.string_types)