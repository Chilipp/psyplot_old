import os.path as osp

test_dir = osp.dirname(__file__)


def get_file(fname):
    """Get the full path to the given file name in the test directory"""
    return osp.join(test_dir, fname)
