"""
This file contains miscellaneous stuff used in the project
"""

import os
import tempfile

_TMP_DIR = tempfile.TemporaryDirectory()
TMP_DIR_PATH = os.path.join(_TMP_DIR.name, 'qgis')
