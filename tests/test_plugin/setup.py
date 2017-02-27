from setuptools import setup, find_packages


setup(name='psyplot_test',
      license="GPLv2",
      packages=find_packages(exclude=['docs', 'tests*', 'examples']),
      entry_points={'psyplot_test': ['plugin=psyplot_test.plugin']},
      zip_safe=False)
