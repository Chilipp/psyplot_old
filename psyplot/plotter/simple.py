import warnings
depr_message = (
    "The psyplot.plotter.simple module has been moved to the "
    "psy_simple.plotters module!")

try:
    from psy_simple.plotters import *
    warnings.warn(depr_message, DeprecationWarning)
except ImportError:
    raise ImportError(depr_message)
