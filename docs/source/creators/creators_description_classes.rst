Model classes and requirements
==============================

=======================
Supported model classes
=======================
The plugin supports the following model classes:
 * Segmentation Model (aka. :code:`Segmentor`)
 * Detection Model (aka. :code:`Detector`)
 * Regression Model (aka. :code:`Regressor`)

Once the processing of ortophoto is finished, a report with model-specific information will be presented.

Common rules for models and processing:
 * Model needs to be in ONNX format, which contains both the network architecture and weights.
 * All model classes process the data in chunks called 'tiles', that is a small part of the entire ortophoto - tiles size and overlap is configurable.
 * Every model should have one input of size :code:`[BATCH_SIZE, CHANNELS, SIZE_PX, SIZE_PX]`. :code:`BATCH_SIZE` can be 1 or dynamic.
 * Size of processed tiles (in pixels) is model defined, but needs to be equal in x and y axes, so that the tiles can be square.
 * If the processed tile needs to be padded (e.g. on otophoto borders) it will be padded with 0 values.
 * Input image data - only uint8_t value for each pixel channel is supported


==================
Segmentation Model
==================
Segmentation models allow to solve problem of Image segmentation, that is assigning a distinct class (category) to each pixel.

Example application is segmenting earth surface areas into the following categories: forest, road, buildings, water, other.

The segmentation model output is also an image, with same dimension as the input tile, but instead of 'CHANNELS' dimension, each output class has a separate image.
Therefore, the shape of model output is :code:`[BATCH_SIZE, NUM_CLASSES, SIZE_PX, SIZE_PX]`.

We support the following types of models:
 * single output (one head) with the following output shapes:
    * :code:`[BATCH_SIZE, 1, SIZE_PX, SIZE_PX]` - one class with sigmoid activation function (binary classification)
    * :code:`[BATCH_SIZE, NUM_CLASSES, SIZE_PX, SIZE_PX]` - multiple classes with softmax activation function (multi-class classification) - outputs sum to 1.0
 * multiple outputs (multiple heads) with each output head composed of the same shapes as single output.

 Metaparameter :code:`class_names` saved in the model file should be as follows in this example:
    * for single output with binary classification (sigmoid): :code:`[{0: "background", 1: "class_name"}]`
    * for single output with multi-class classification (softmax): :code:`[{0: "class0", 1: "class1", 2: "class2"}]` or :code:`{0: "class0", 1: "class1", 2: "class2"}`
    * for multiple outputs (multiple heads): :code:`[{0: "class0", 1: "class1", 2: "class2"}, {0: "background", 1: "class_name"}]`

Output report contains information about percentage coverage of each class.


===============
Detection Model
===============
Detection models allow to solve problem of objects detection, that is finding an object of predefined class in the image, and marking with a bound box (rectangle).

Example application is detection of oil and water tanks on satellite images.

The detection model output is list of bounding boxes, with assigned class and confidence value. This information is not really standardized between different model architectures.
Currently plugin supports :code:`YOLOv5`, :code:`YOLOv6`, :code:`YOLOv7`, :code:`YOLOv9` and :code:`ULTRALYTICS` output types. Detection model also supports the instance segmentation output type from :code:`ULTRALYTICS`.

For each object class, a separate vector layer can be created, with information saved as rectangle polygons (so the output can be potentially easily exported to a text).

Output report contains information about detections, that is how many objects of each class has been detected.

================
Regression Model
================
Regression models allow to solve problem of Regression Analysis, that is assigning a non-distinct (continuous) value to each pixel.

Example application is determining the moisture content in soil, as percentage from 0.0 to 100.0 %, with an individual value assigned to each pixel.

The segmentation model output is also an image, with same dimension as the input tile, with one or many output maps. Each output map contains the values for pixels.
Therefore, the shape of model output is :code:`[BATCH_SIZE, NUMBER_OF_OUTPUT_MAPS, SIZE_PX, SIZE_PX]`.

One output layer will be created for each output map (channel).
For each output, a raster layer will be created, where each pixel has individual value assigned.
Usually, only one output map (class) is used, as the model usually tries to solve just one task.

Output report contains statistics for each class, that is average, min, max and standard deviation of values.

=====================
SuperResolution Model
=====================
SuperResolution models allow to solve problem of increasing the resolution of the image. The model takes a low resolution image as input and outputs a high resolution image.

Example application is increasing the resolution of satellite images.

The superresolution model output is also an image, with same dimension as the input tile, but with higher resolution (GDS).

================
Extending Models
================

For extending the functionality and adding new model classes, please visit section `Extenging plugin functionality`.


