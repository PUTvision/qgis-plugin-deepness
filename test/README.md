# Developing and running test
To run unit tests issue the following commands:
```
export PYTHONPATH=$PYTHONPATH:`pwd`/src
export PYTHONPATH=$PYTHONPATH:`pwd`
python3 -m pytest --cov=plugin/deep_segmentation_framework/ --cov-report html test/
xdg-open htmlcov/index.html
```

## pip packages issues
There is some conflict between `opencv-python` and `PyQt5`.
To be able to run tests you may need to execute:
```
pip uninstall -y opencv-python && pip install opencv-python-headless
```

Once done with testing, you may need to restore the previous package:
```
pip uninstall -y opencv-python-headless && pip install opencv-python
```

## testing file
Some testing is performed on one file with fotomap, to have the same processing as in the QGis GUI.
These tests are hardcoded to use this file specifically.
