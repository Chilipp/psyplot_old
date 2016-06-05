from argparse import ArgumentParser
import inspect
from psyplot.docstring import docstrings
from psyplot.compat.pycompat import OrderedDict


class FuncArgParser(ArgumentParser):
    """Subclass of an argument parser that get's parts of the information
    from a given function"""

    def __init__(self, *args, **kwargs):
        super(FuncArgParser, self).__init__(*args, **kwargs)
        self.__arguments = OrderedDict()
        self.__funcs = []
        self.__main = None

    def setup_args(self, func):
        """Add the parameters from the given `func` to the parameter settings
        """
        self.__funcs.append(func)
        args_dict = self.__arguments
        args, varargs, varkw, defaults = inspect.getargspec(func)
        full_doc = inspect.getdoc(func)
        doc = docstrings._get_section(full_doc, 'Parameters') + '\n'
        doc += docstrings._get_section(full_doc, 'Other Parameters')
        doc = doc.rstrip()
        default_min = len(args) - len(defaults)
        for i, arg in enumerate(args):
            if arg == 'self' or arg in args_dict:
                continue
            arg_doc = docstrings._keep_params(doc, [arg]) or \
                docstrings._keep_types(doc, [arg])
            args_dict[arg] = d = {'dest': arg, 'short': arg, 'long': arg}
            if arg_doc:
                lines = arg_doc.splitlines()
                d['help'] = '\n'.join(lines[1:])
                metavar = lines[0].split(':', 1)
                if i >= default_min:
                    d['default'] = defaults[i - default_min]
                if len(metavar) > 1:
                    dtype = metavar[1].strip()
                    if dtype == 'bool' and 'default' in d:
                        d['action'] = 'store_false' if d['default'] else \
                            'store_true'
                    else:
                        d['metavar'] = metavar[1].strip()
                else:
                    d['positional'] = True

    def update_arg(self, arg, if_existent=True, **kwargs):
        """Update the `add_argument` data for the given parameter
        """
        if not if_existent:
            self.__arguments.setdefault(arg, kwargs)
        self.__arguments[arg].update(kwargs)

    def pop_key(self, arg, key, *args, **kwargs):
        """Delete a previously defined key for the `add_argument`
        """
        return self.__arguments[arg].pop(key, *args, **kwargs)

    def create_arguments(self):
        """Create and add the arguments"""
        ret = []
        for arg, d in self.__arguments.items():
            try:
                is_positional = d.pop('positional', False)
                short = d.pop('short')
                long_name = d.pop('long', None)
                if short == long_name:
                    long_name = None
                args = [short, long_name] if long_name else [short]
                if not is_positional:
                    for i, arg in enumerate(args):
                        args[i] = '-' * (i + 1) + arg
                else:
                    d.pop('dest', None)
                group = d.pop('group', self)
                ret.append(group.add_argument(*args, **d))
            except Exception:
                print('Error while creating argument %s' % arg)
                raise
        return ret

    @docstrings.get_sectionsf('FuncArgParser.parse_to_func')
    @docstrings.dedent
    def parse_to_func(self, args=None):
        """
        Parse the given arguments to the main function

        Parameters
        ----------
        args: list
            The list of arguments given to the
            :meth:`ArgumentParser.parse_args` function. If None, the sys.argv
            is used."""
        if args is not None:
            (self.__main or self.__funcs[0])(**vars(self.parse_args(args)))
        else:
            (self.__main or self.__funcs[0])(**vars(self.parse_args()))

    @docstrings.dedent
    def parse_known_to_func(self, args=None):
        """
        Parse the known arguments from the given to the main function

        Parameters
        ----------
        %(FuncArgParser.parse_to_func.parameters)s"""
        if args is not None:
            (self.__main or self.__funcs[0])(
                **vars(self.parse_known_args(args)[0]))
        else:
            (self.__main or self.__funcs[0])(
                **vars(self.parse_known_args()[0]))

    def set_main(self, func):
        """Set the function that is called by the :meth:`parse_to_func` and
        :meth:`parse_known_to_func` function"""
        self.__main = func
