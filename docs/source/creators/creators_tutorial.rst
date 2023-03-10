Model creation tutorial
=======================


=========
Detection
=========

For one of models in our zoo - specifically for cars detection on aerial images - a complete tutorial is provided in a jupyter notebook:

  .. code-block::

        ./tutorials/detection/cars_yolov7/car_detection__prepare_and_train.ipynb


The notebook covers:
 * downloading yolov7 repository
 * downloading the training dataset
 * preparing training data and labels in yolov7 format
 * running th training and testing
 * conversion to ONNX model
 * adding default parameters for Deepness plugin

Example model inference can be found in the :code:`Examples` section.
