#!/usr/bin/env python
"""Script to create the reference figures for the psyplot tests

This script may be used from the command line run all tests from the psyplot
package. See::

    python create_references.py -h

for details.
"""

import _base_testing as bt
import os
import glob
import sys

argv = list(sys.argv)
if '-r' not in argv and '--ref' not in argv:
    argv += ['-r']

files = glob.glob(os.path.join(bt.test_dir, 'test_*.py'))

for f in files:
    bt.RefTestProgram(os.path.splitext(os.path.basename(f))[0], exit=False,
                      argv=tuple(argv))
