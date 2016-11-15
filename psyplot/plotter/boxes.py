import warnings
depr_message = (
    "The psyplot.plotter.boxes module has been moved to the psy_maps.boxes "
    "module!")

try:
    from psy_maps.boxes import *
    warnings.warn(depr_message, DeprecationWarning)
except ImportError:
    raise ImportError(depr_message)
