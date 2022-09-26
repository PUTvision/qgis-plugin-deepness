import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_icon_path():
    return get_image_path('icon.png')


def get_image_path(image_name):
    return os.path.join(SCRIPT_DIR, image_name)