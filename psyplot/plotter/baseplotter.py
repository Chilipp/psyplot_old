import warnings
depr_message = (
    "The baseplotter module has been moved to the psy_simple.baseplotter "
    "module!")

try:
    from psy_simple.baseplotter import *
    warnings.warn(depr_message, DeprecationWarning)
except ImportError:
    raise ImportError(depr_message)
