import datetime as dt
import logging as _logging
from psyplot.warning import warn, critical, disable_warnings
from psyplot.config.rcsetup import rcParams
import psyplot.config as config
from psyplot.data import (
    ArrayList, InteractiveArray, InteractiveList, open_dataset, open_mfdataset)

__version__ = "0.2.17"
__author__ = "Philipp Sommer (philipp.sommer@unil.ch)"

logger = _logging.getLogger(__name__)
logger.debug(
    "%s: Initializing psyplot, version %s",
    dt.datetime.now().isoformat(), __version__)
logger.debug("Logging configuration file: %s", config.logcfg_path)
logger.debug("Configuration file: %s", config.config_path)


rcParams.HEADER += "\n\npsyplot version: " + __version__
