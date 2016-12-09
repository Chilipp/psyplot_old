v1.0.0
======

Added
-----
* Changelog

Changed
-------
* The modules in the psyplot.plotter modules have been moved to separate
  packages to make the debugging and testing easier

  - The psyplot.plotter.simple, baseplotter and colors modules have been moved
    to the psy-simple package
  - The psyplot.plotter.maps and boxes modules have been moved to the psy-maps
    package
  - The psyplot.plotter.linreg module has been moved to the psy-fit package
* The endings of the yaml configuration files are now all *.yml*. Hence,

  - the configuration file name is now *psyplotrc.yml* instead of
    *psyplotrc.yaml*
  - the default logging configuration file name is now *logging.yml* instead
    of *logging.yaml*
