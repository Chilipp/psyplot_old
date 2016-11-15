import warnings
depr_message = (
    "The psyplot.plotter.maps module has been moved to the psy_maps.plotters "
    "module!")

try:
    from psy_maps.plotters import *
    warnings.warn(depr_message, DeprecationWarning)
except ImportError:
    raise ImportError(depr_message)
