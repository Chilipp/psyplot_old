import warnings
depr_message = (
    "The psyplot.plotter.colors module has been moved to the "
    "psy_simple.colors module!")

try:
    from psy_simple.colors import *
    warnings.warn(depr_message, DeprecationWarning)
except ImportError:
    raise ImportError(depr_message)
