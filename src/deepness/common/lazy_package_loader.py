"""
This file contains utility to lazy import packages
"""

import importlib


class LazyPackageLoader:
    """ Allows to wrap python package into a lazy version, so that the package will be loaded once it is actually used

    Usage:
        cv2 = LazyPackageLoader('cv2')  # This will not import cv2 yet
        ...
        cv2.waitKey(3)  # here will be the actual import
    """

    def __init__(self, package_name):
        self._package_name = package_name
        self._package = None

    def __getattr__(self, name):
        if self._package is None:
            self._package = importlib.import_module(self._package_name)
        return getattr(self._package, name)
