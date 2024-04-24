Training Data Export Tool
=========================

Apart from model inference functionality, the plugin contains :code:`Training Data Export Tool`.
This tool allows to prepare the following data for the model training process:
 - specified part of ortophoto - divided into tiles (each tile saved as a separate file)
 - specified part of the annotation layer, which can be used as ground-truth for model training. Each mask tile corresponds to one ortophoto tile.

Exported data follows the same rules as inference, that is the user needs to specify layer and what part of thereof should be processed.
Tile size and overlap between consecutive tiles is also configurable.

Approach with exporting mask tiles is mostly helpful for segmentation problems, because one can create complete set of training data within QGIS, by drawing proper segmentation polygons.

Tool implemented in :code:`training_data_export_widget.py` and :code:`map_processor_training_data_export.py`.
