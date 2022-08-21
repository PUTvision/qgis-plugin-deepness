"""pos[0]-offset, pos[1]-offset,pos[0]+offset,pos[1]+offset)
This test requires file 'borecko/fotomapa.tif'.
If you do not have this file, you would need to update a few parameters in these tests.

In the future, consider using a generated file instead...
"""

import sys
from unittest.mock import MagicMock

import pytest
from qgis.PyQt.QtWidgets import QApplication
from qgis.core import QgsCoordinateReferenceSystem, QgsRectangle, QgsApplication
from qgis.core import QgsRasterLayer

from deep_segmentation_framework.common.inference_parameters import ProcessedAreaType
from deep_segmentation_framework.deep_segmentation_framework_dockwidget import DeepSegmentationFrameworkDockWidget
from deep_segmentation_framework.processing.map_processor import MapProcessor


RASTER_FILE_PATH = '/home/przemek/Desktop/corn/borecko/fotomapa.tif'
RASTER_FILE_PATH = '/home/przemek/Desktop/corn/borecko/fake_fotomap.tif'
# RASTER_FILE_PATH = '/home/przemek/Desktop/corn/10ha_copy/fotomapa.tif'

PROCESSED_EXTENT_1 = QgsRectangle(  # big part of the field (15 tiles with 512px)
        638895.87214042595587671, 5802472.81716971844434738,
        638973.46824810293037444, 5802515.99556608032435179)
PROCESSED_EXTENT_2_FULL = QgsRectangle(  # entire field
        638838.69500850629992783, 5802263.68493685312569141,
        639034.16520346351899207, 5802604.9122637296095490)
PROCESSED_EXTENT_3_SINGLE_TILE_WITH_BORDER = QgsRectangle(
        638923.81203882768750191, 5802448.52505646646022797,
        638963.58058058377355337, 5802469.66866450011730194)
PROCESSED_EXTENT_4_FAKE_FOTOMAPA = QgsRectangle(
        638904.991, 5802_482.419,
        638968.477, 5802_510.616)


def dummy_test():
    assert 1 + 1 == 2


def generic_processing_test():
    qgs = QgsApplication([b''], False)
    qgs.setPrefixPath('/usr/bin/qgis', True)
    qgs.initQgis()

    dockwidget = DeepSegmentationFrameworkDockWidget(iface=None)
    inference_parameters = dockwidget.get_inference_parameters()
    inference_parameters.model_file_path = '/home/przemek/Desktop/corn/model/corn_segmentation_model.onnx'
    processed_extent = PROCESSED_EXTENT_4_FAKE_FOTOMAPA
    MapProcessor.ModelWrapper = MagicMock()

    # we want to use a fake extent, which is the Visible Part of the map
    # and we need to mock it's function calls
    inference_parameters.processed_area_type = ProcessedAreaType.VISIBLE_PART
    map_canvas = MagicMock()
    map_canvas.extent = lambda: processed_extent
    map_canvas.mapSettings().destinationCrs = lambda: QgsCoordinateReferenceSystem("EPSG:32633")

    rlayer = QgsRasterLayer(RASTER_FILE_PATH, 'fotomap')
    if rlayer.width() == 0:
        raise Exception("0 width - rlayer not loaded properly. Probably invalid file path?")

    rlayer.setCrs(QgsCoordinateReferenceSystem("EPSG:32633"))

    map_processor = MapProcessor(
        rlayer=rlayer,
        map_canvas=map_canvas,
        inference_parameters=inference_parameters,
    )

    map_processor.run()


if __name__ == '__main__':
    dummy_test()
    generic_processing_test()
    print('Done')
