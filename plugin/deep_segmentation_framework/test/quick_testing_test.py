"""pos[0]-offset, pos[1]-offset,pos[0]+offset,pos[1]+offset)
This test requires file 'borecko/fotomapa.tif'.
If you do not have this file, you would need to update a few parameters in these tests.

In the future, consider using a generated file instead...
"""

import sys
from unittest.mock import MagicMock

import pytest
from qgis.PyQt.QtWidgets import QApplication
from qgis.core import QgsProject
from qgis.core import QgsVectorLayer
from qgis.core import QgsCoordinateReferenceSystem, QgsRectangle, QgsApplication
from qgis.core import QgsRasterLayer

from deep_segmentation_framework.common.inference_parameters import ProcessedAreaType
from deep_segmentation_framework.deep_segmentation_framework_dockwidget import DeepSegmentationFrameworkDockWidget
from deep_segmentation_framework.processing.map_processor import MapProcessor


RASTER_FILE_PATH = '/home/przemek/Desktop/corn/borecko/fotomapa.tif'
MODEL_FILE_PATH = '/home/przemek/Desktop/corn/corn_segmentation_model.onnx'


def get_rlayer():
    rlayer = QgsRasterLayer(RASTER_FILE_PATH, 'fotomap')
    if rlayer.width() == 0:
        raise Exception("0 width - rlayer not loaded properly. Probably invalid file path?")
    rlayer.setCrs(QgsCoordinateReferenceSystem("EPSG:32633"))
    QgsProject.instance().addMapLayer(rlayer)
    return rlayer


def init_qgis():
    qgs = QgsApplication([b''], False)
    qgs.setPrefixPath('/usr/bin/qgis', True)
    qgs.initQgis()
    return qgs


def main():
    # TODO - prepare inference parameters manually

    qgs = init_qgis()
    rlayer = get_rlayer()

    dockwidget = DeepSegmentationFrameworkDockWidget(iface=MagicMock())
    dockwidget._model_wrapper = MagicMock()
    dockwidget._model_wrapper.process = lambda x: x[:, :, 0]

    dockwidget.lineEdit_modelPath.text = lambda: MODEL_FILE_PATH

    inference_parameters = dockwidget.get_inference_parameters()
    inference_parameters.processed_area_type = ProcessedAreaType.ENTIRE_LAYER

    map_processor = MapProcessor(
        rlayer=rlayer,
        vlayer_mask=None,
        map_canvas=None,
        inference_parameters=inference_parameters,
    )

    # map_processor.run()



if __name__ == '__main__':
    main()
    print('Done')
