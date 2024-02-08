Adding metadata to ONNX model
=============================

The plugin allows you to load the meta parameters of the onnx model automatically. Predefined parameter types are designed for the simplicity of the user's work. The written metadata is especially important for users who are not familiar with the technical details.


===================================
List of parameters parsed by plugin
===================================

+----------------------+-------+---------------------------------------+-------------------------------------------------------------+
| Parameter            |  Type |            Example                    | Description                                                 |
+======================+=======+=======================================+=============================================================+
| model_type           |  str  |   :code:`'Segmentor'`                 | Types of models available: Segmentor, Regressor, Detector.  |
+----------------------+-------+---------------------------------------+-------------------------------------------------------------+
| class_names          |  dict | :code:`{0: 'background', 1: 'field'}` | A dictionary that maps a class id to its name.              |
+----------------------+-------+---------------------------------------+-------------------------------------------------------------+
| resolution           | float |        :code:`100`                    | Real-world resolution of images (centimeters per pixel).    |
+----------------------+-------+---------------------------------------+-------------------------------------------------------------+
| tiles_size           |  int  |        :code:`512`                    | What size (in pixels) is the tile to crop.                  |
+----------------------+-------+---------------------------------------+-------------------------------------------------------------+
| tiles_overlap        |  int  |         :code:`40`                    | How many percent of the image size overlap.                 |
+----------------------+-------+---------------------------------------+-------------------------------------------------------------+
| standardization_mean |  list |         :code:`[0.0, 0.0, 0.0]`       | Mean - if you want to standarize input after normalisation  |
+----------------------+-------+---------------------------------------+-------------------------------------------------------------+
| standardization_std  |  list |         :code:`[1.0, 1.0, 1.0]`       | Std - if you want to standarize input after normalisation   |
+----------------------+-------+---------------------------------------+-------------------------------------------------------------+
| seg_thresh           | float |       :code:`0.5`                     | Segmentor: class confidence threshold.                      |
+----------------------+-------+---------------------------------------+-------------------------------------------------------------+
| seg_small_segment    |  int  |       :code:`7`                       | Segmentor: remove small occurrences of the class.           |
+----------------------+-------+---------------------------------------+-------------------------------------------------------------+
| reg_output_scaling   | float |       :code:`1.0`                     | Regressor: scaling factor for the model output.             |
+----------------------+-------+---------------------------------------+-------------------------------------------------------------+
| det_conf             | float |       :code:`0.6`                     | Detector: object confidence threshold.                      |
+----------------------+-------+---------------------------------------+-------------------------------------------------------------+
| det_iou_thresh       | float |       :code:`0.4`                     | Detector: IOU threshold for NMS.                            |
+----------------------+-------+---------------------------------------+-------------------------------------------------------------+
| det_type             | str   |       :code:`YOLO_v5_or_v7_default`   | Detector: type of the detector model format                 |
+----------------------+-------+---------------------------------------+-------------------------------------------------------------+
| det_remove_overlap   | bool  |       :code:`True`                    | Detector: Whether overlapping detection should be removed   |
+----------------------+-------+---------------------------------------+-------------------------------------------------------------+

=======
Example
=======

The example below shows how to add string, float, and dictionary metadata into a model. Note that metadata is created while :code:`model.metadata_props.add()` is called. Moreover, the metadata value has to be a byte type.

.. code-block::

    import json
    import onnx

    model = onnx.load('deeplabv3_landcover_4c.onnx')

    class_names = {
        0: '_background',
        1: 'building',
        2: 'woodland',
        3: 'water',
        4: 'road',
    }

    m1 = model.metadata_props.add()
    m1.key = 'model_type'
    m1.value = json.dumps('Segmentor')

    m2 = model.metadata_props.add()
    m2.key = 'class_names'
    m2.value = json.dumps(class_names)

    m3 = model.metadata_props.add()
    m3.key = 'resolution'
    m3.value = json.dumps(50)

    # optional, if you want to standarize input after normalisation
    m4 = model.metadata_props.add()
    m4.key = 'standardization_mean'
    m4.value = json.dumps([0.0, 0.0, 0.0])

    m5 = model.metadata_props.add()
    m5.key = 'standardization_std'
    m5.value = json.dumps([1.0, 1.0, 1.0])

    onnx.save(model, 'deeplabv3_landcover_4c.onnx')


You can also use a script setting the most important parameters in the plugin repository (file :code:`tools/add_model_metadata.py`).
