"""Logging configuration module of the psyplot package

This module defines the essential functions for setting up the
:class:`logging.Logger` instances that are used by the psyplot package."""
import os
import logging
import logging.config
import yaml
from ..docstring import dedent


@dedent
def setup_logging(default_path=None, default_level=logging.INFO,
                  env_key='LOG_PSYPLOT'):
    """
    Setup logging configuration

    Parameters
    ----------
    default_path: str
        Default path of the yaml logging configuration file. If None, it
        defaults to the 'logging.yaml' file in the config directory
    default_level: int
        Default: :data:`logging.INFO`. Default level if default_path does not
        exist
    env_key: str
        environment variable specifying a different logging file than
        `default_path` (Default: 'LOG_CFG')

    Returns
    -------
    path: str
        Path to the logging configuration file

    Notes
    -----
    Function taken from
    http://victorlin.me/posts/2012/08/26/good-logging-practice-in-python"""
    from .rcsetup import _get_home
    path = default_path or os.path.dirname(__file__) + '/logging.yaml'
    value = os.getenv(env_key, None)
    home = _get_home()
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.load(f.read())
        for handler in config.get('handlers', {}).values():
            if '~' in handler.get('filename', ''):
                handler['filename'] = handler['filename'].replace(
                    '~', _get_home())
        logging.config.dictConfig(config)
    else:
        path = None
        logging.basicConfig(level=default_level)
    return path
