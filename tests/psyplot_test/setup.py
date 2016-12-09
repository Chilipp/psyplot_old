from setuptools import setup, find_packages
import sys


setup(name='psyplot_test',
      license="GPLv2",
      packages=find_packages(exclude=['docs', 'tests*', 'examples']),
      entry_points={'psyplot_test': ['rcParams=psyplot_test.plugin:rcParams']},
      zip_safe=False)
