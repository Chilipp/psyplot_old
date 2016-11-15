import warnings
depr_message = (
    "The psyplot.plotter.linreg module has been moved to the psy_fit.plotters "
    "module!")

try:
    from psy_fit.plotters import *
    warnings.warn(depr_message, DeprecationWarning)
except ImportError:
    raise ImportError(depr_message)
