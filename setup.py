from setuptools import setup, find_packages
import sys

needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='psyplot',
      version='0.0.2.dev1',
      description='Python package for interactive data visualization',
      long_description=readme(),
      classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Operating System :: OS Independent',
      ],
      keywords='visualization netcdf raster cartopy earth-sciences',
      url='https://github.com/Chilipp/psyplot',
      author='Philipp Sommer',
      author_email='philipp.sommer@unil.ch',
      license="GPLv2",
      packages=find_packages(exclude=['docs', 'tests*']),
      install_requires=[
          'matplotlib',
          'xray>=0.6',
          'PyYAML'
      ],
      setup_requires=pytest_runner,
      tests_require=['pytest'],
      include_package_data=True,
      zip_safe=False)
