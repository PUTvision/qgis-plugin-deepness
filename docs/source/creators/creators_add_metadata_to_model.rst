How to add metadata to onnx model
=================================

The plugin allows you to load the meta parameters of the onnx model automatically. Predefined parameter types are designed for the simplicity of the user's work. The written metadata is especially important for users who are not familiar with the technical details.


====================================
List of parameters parsed by plugin
====================================

+--------------------+-------+---------------------------------------+-------------------------------------------------------------+
| Parameter          |  Type |            Example                    | Description                                                 |
+====================+=======+=======================================+=============================================================+
| model_type         |  str  |   :code:`'segmenter'`                 | Types of models available: segmenter, regressor, detector.  |
+--------------------+-------+---------------------------------------+-------------------------------------------------------------+
| class_names        |  dict | :code:`{0: 'background', 1: 'field'}` | A dictionary that maps a class id to its name.              |
+--------------------+-------+---------------------------------------+-------------------------------------------------------------+
| resolution         | float |        :code:`100`                    |                                                             |
+--------------------+-------+---------------------------------------+-------------------------------------------------------------+
| tiles_size         |  int  |        :code:`512`                    |                                                             |
+--------------------+-------+---------------------------------------+-------------------------------------------------------------+
| tiles_overlap      |  int  |         :code:`40`                    |                                                             |
+--------------------+-------+---------------------------------------+-------------------------------------------------------------+
| seg_argmax         |  bool |      :code:`False`                    |                                                             |
+--------------------+-------+---------------------------------------+-------------------------------------------------------------+
| seg_thresh         | float |       :code:`0.5`                     |                                                             |
+--------------------+-------+---------------------------------------+-------------------------------------------------------------+
| seg_small_segment  |  int  |       :code:`7`                       |                                                             |
+--------------------+-------+---------------------------------------+-------------------------------------------------------------+
| reg_output_scaling | float |       :code:`1.0`                     |                                                             |
+--------------------+-------+---------------------------------------+-------------------------------------------------------------+
| det_conf           | float |       :code:`0.6`                     |                                                             |
+--------------------+-------+---------------------------------------+-------------------------------------------------------------+
| det_iou_thresh     | float |       :code:`0.4`                     |                                                             |
+--------------------+-------+---------------------------------------+-------------------------------------------------------------+


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
    m1.value = json.dumps('segmenter')

    m2 = model.metadata_props.add()
    m2.key = 'class_names'
    m2.value = json.dumps(class_names)

    m3 = model.metadata_props.add()
    m3.key = 'resolution'
    m3.value = json.dumps(50)

    onnx.save(model, 'deeplabv3_landcover_4c.onnx')
