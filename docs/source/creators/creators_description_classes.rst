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
 * Every model should have one input of size :code:`[BATCH_SIZE, CHANNELS, SIZE_PX, SIZE_PX]`. :code:`BATCH_SIZE` can be 1.
 * Size of processed tiles (in pixels) is model defined, but needs to be equal in x and y axes, so that the tiles can be square.
 * If the processed tile needs to be padded (e.g. on otophoto borders) it will be padded with 0 values.
 * Input image data - only uint8_t value for each pixel channel is supported


==================
Segmentation Model
==================
Segmentation models allow to solve problem of Image segmentation, that is assigning a distinct class (category) to each pixel.

Example application is segmenting earth surface areas into the following categories: forest, road, buildings, water, other.

The segmentation model output is also an image, with same dimension as the input tile, but instead of 'CHANNELS' dimension, each output class has a separate image.
Therefore, the shape of model output is :code:`[BATCH_SIZE, NUM_CLASSES, SIZE_PX, SIZE_PX)`.

For each output class, a separate vector layer can be created.

Output report contains information about percentage coverage of each class.


===============
Detection Model
===============
Detection models allow to solve problem of objects detection, that is finding an object of predefined class in the image, and marking with a bound box (rectangle).

Example application is detection of oil and water tanks on satellite images.

The detection model output is list of bounding boxes, with assigned class and confidence value. This information is not really standardized between different model architectures.
Currently plugin supports :code:`YOLOv5` and :code:`YOLOv7` output types.

For each object class, a separate vector layer can be created, with information saved as rectangle polygons (so the output can be potentially easily exported to a text).

Output report contains information about detections, that is how many objects of each class has been detected.

================
Regression Model
================
Regression models allow to solve problem of Regression Analysis, that is assigning a non-distinct (continuous) value to each pixel.

Example application is determining the moisture content in soil, as percentage from 0.0 to 100.0 %, with an individual value assigned to each pixel.

The segmentation model output is also an image, with same dimension as the input tile, with one or many output maps. Each output map contains the values for pixels.
Therefore, the shape of model output is :code:`[BATCH_SIZE, NUMBER_OF_OUTPUT_MAPS, SIZE_PX, SIZE_PX)`.

One output layer will be created for each output map (channel).
For each output, a raster layer will be created, where each pixel has individual value assigned.
Usually, only one output map (class) is used, as the model usually tries to solve just one task.

Output report contains statistics for each class, that is average, min, max and standard deviation of values.


================
Extending Models
================
For extending the functionality and adding new model classes, please visit section `Extenging plugin functionality`.


