import datetime as dt
import logging as _logging
from .warning import warn, critical, disable_warnings
from .config.rcsetup import rcParams
from .data import (
    ArrayList, InteractiveArray, InteractiveList, open_dataset, open_mfdataset)

__version__ = "0.00b"
__author__ = "Philipp Sommer (philipp.sommer@studium.uni-hamburg.de)"


logger = _logging.getLogger(__name__)
logger.debug(
    "%s: Initializing psyplot, version %s",
    dt.datetime.now().isoformat(), __version__)
logger.debug("Logging configuration file: %s", config.logcfg_path)
logger.debug("Configuration file: %s", config.config_path)
