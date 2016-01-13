"""Script to create reference pictures for the testing of the psyplot package

This script may be used from the command line to create the required reference
figures for testing the psyplot package. For help on the parameters see::

    python create_references.py -h
"""

from _base_testing import RefTestProgram
import sys
argv = list(sys.argv)
if '-r' not in argv or '--ref' not in argv:
    argv = argv[:1] + ['-r'] + argv[1:] if len(argv) > 1 else []
RefTestProgram('test_maps', argv=argv)
RefTestProgram('test_simpleplotter', argv=argv)
