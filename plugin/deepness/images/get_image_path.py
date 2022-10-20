"""
This file contains image related functionalities
"""

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_icon_path() -> str:
    """ Get path to the file with the main plugin icon

    Returns
    -------
    str
        Path to the icon
    """
    return get_image_path('icon.png')


def get_image_path(image_name) -> str:
    """ Get path to an image resource, accessing it just by the name of the file (provided it is in the common directory)

    Returns
    -------
    str
        file path
    """
    return os.path.join(SCRIPT_DIR, image_name)
