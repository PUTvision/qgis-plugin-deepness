Unit testing
============

In order to simplify development and debugging, the code is designed as modular and testable.
Unit tests can be found in :code:`./test` directory of the main repository.

Unit tests, apart from being automated tests, allow to easily develop and debug the code (e.g. in PyCharm or vs code),
without the need for interaction in QGIS GUI application. Also the UI module is vaguely tested -
at least to confirm the code is runnable, with some basic checks.

As part of the unit tests, exemplary models and ortophotos are used.

In order to run unit tests, please consult the :code:`test/README.md` in the main repository directory.
