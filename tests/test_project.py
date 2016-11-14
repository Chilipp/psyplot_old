"""Test module of the :mod:`psyplot.project` module"""
import os
import unittest
import _base_testing as bt


def get_file(fname):
    return os.path.join(bt.test_dir, fname)


class ProjectTester(unittest.TestCase):

    plot_type = 'project'

    def test_save_and_load(self):
        """Test project reproducability through the save and load method"""
        pass


if __name__ == '__main__':
    unittest.main()
