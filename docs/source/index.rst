====
Home
====

.. note::

   This project is under active development.


Plugin is aimed as a tool for casual QGIS users, which don't need to be familiar with Machine Learning.
The plugin packs all the complexity behind a simple UI, so that the user can easily process their data with the power of Deep Neural Networks.

Of course, the model needs to be first created by a user familiar with Machine Learning, according to model requirements.
Nevertheless, the entire complexity of processing is implemented in the plugin.


Home
----

.. toctree::
   
   self
   license

.. toctree::
   :maxdepth: 1
   :caption: For Users in QGis

   main/main_features
   main/main_supported_versions
   main/main_installation
   main/main_ui_explanation
   main/main_ui_video
   main/main_model_zoo

.. toctree::
   :maxdepth: 1
   :caption: Examples

   example/example_segmentation_landcover
   example/example_detection_planes_yolov7
   example/example_detection_oils_yolov5
   example/example_regression_vegetation_index_rgb

.. toctree::
   :maxdepth: 1
   :caption: For Model Creators

   creators/creators_description_classes
   creators/creators_export_training_data_tool
   creators/creators_example_onnx_model
   creators/creators_add_metadata_to_model

.. toctree::
   :maxdepth: 1
   :caption: For Developers

   dev/dev_general_info
   dev/dev_extending_plugin
   dev/dev_unit_tests
   dev/dev_api
