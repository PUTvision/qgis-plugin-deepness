"""pos[0]-offset, pos[1]-offset,pos[0]+offset,pos[1]+offset)
This test requires file 'borecko/fotomapa.tif'.
If you do not have this file, you would need to update a few parameters in these tests.

In the future, consider using a generated file instead...
"""

import sys
from unittest.mock import MagicMock

import pytest
from qgis.PyQt.QtWidgets import QApplication
from qgis.core import QgsVectorLayer, QgsProject
from qgis.core import QgsCoordinateReferenceSystem, QgsRectangle, QgsApplication
from qgis.core import QgsRasterLayer

from deep_segmentation_framework.common.inference_parameters import ProcessedAreaType
from deep_segmentation_framework.deep_segmentation_framework_dockwidget import DeepSegmentationFrameworkDockWidget
from deep_segmentation_framework.processing.map_processor import MapProcessor


RASTER_FILE_PATH = '/home/przemek/Desktop/corn/borecko/fotomapa.tif'
# RASTER_FILE_PATH = '/home/przemek/Desktop/corn/borecko/fake_fotomap.tif'
# RASTER_FILE_PATH = '/home/przemek/Desktop/corn/10ha_copy/fotomapa.tif'

VLAYER_MASK_FILE_PATH = '/home/przemek/Desktop/corn/xxx_vlayer.gpkg'
# VLAYER_MASK_FILE_PATH = '/home/przemek/Desktop/corn/borecko/red_area.gpkg'


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


def get_rlayer():
    rlayer = QgsRasterLayer(RASTER_FILE_PATH, 'fotomap')
    if rlayer.width() == 0:
        raise Exception("0 width - rlayer not loaded properly. Probably invalid file path?")
    rlayer.setCrs(QgsCoordinateReferenceSystem("EPSG:32633"))
    QgsProject.instance().addMapLayer(rlayer)
    return rlayer


def get_vlayer():
    vlayer = QgsVectorLayer(VLAYER_MASK_FILE_PATH)
    if not vlayer.isValid():
        raise Exception("Invalid vlayer! Probably invalid file path?")
    QgsProject.instance().addMapLayer(vlayer)
    return vlayer


def init_qgis():
    qgs = QgsApplication([b''], False)
    qgs.setPrefixPath('/usr/bin/qgis', True)
    qgs.initQgis()
    return qgs


def generic_processing_test__specified_extent():
    # TODO - prepare inference parameters manually

    qgs = init_qgis()

    rlayer = get_rlayer()
    dockwidget = DeepSegmentationFrameworkDockWidget(iface=MagicMock())
    dockwidget._model_wrapper = MagicMock()
    dockwidget._model_wrapper.process = lambda x: x[:, :, 0]
    dockwidget._model_wrapper.get_number_of_channels = lambda: 3
    dockwidget._input_channels_mapping_widget.set_model(dockwidget._model_wrapper)
    dockwidget._rlayer_updated()

    inference_parameters = dockwidget.get_inference_parameters()
    processed_extent = PROCESSED_EXTENT_4_FAKE_FOTOMAPA

    # we want to use a fake extent, which is the Visible Part of the map
    # and we need to mock it's function calls
    inference_parameters.processed_area_type = ProcessedAreaType.VISIBLE_PART
    map_canvas = MagicMock()
    map_canvas.extent = lambda: processed_extent
    map_canvas.mapSettings().destinationCrs = lambda: QgsCoordinateReferenceSystem("EPSG:32633")

    rlayer = get_rlayer()

    map_processor = MapProcessor(
        rlayer=rlayer,
        vlayer_mask=None,
        map_canvas=map_canvas,
        inference_parameters=inference_parameters,
    )

    map_processor.run()


def generic_processing_test__specified_vlayer_mask():
    # TODO - prepare inference parameters manually

    qgs = init_qgis()

    rlayer = get_rlayer()
    vlayer_mask = get_vlayer()
    vlayer_mask.setCrs(rlayer.crs())

    dockwidget = DeepSegmentationFrameworkDockWidget(iface=MagicMock())
    dockwidget._model_wrapper = MagicMock()
    dockwidget._model_wrapper.process = lambda x: x[:, :, 0]
    dockwidget._model_wrapper.get_number_of_channels = lambda: 3
    dockwidget._input_channels_mapping_widget.set_model(dockwidget._model_wrapper)
    dockwidget._rlayer_updated()

    inference_parameters = dockwidget.get_inference_parameters()
    inference_parameters.processed_area_type = ProcessedAreaType.FROM_POLYGONS

    map_processor = MapProcessor(
        rlayer=rlayer,
        vlayer_mask=vlayer_mask,
        map_canvas=None,
        inference_parameters=inference_parameters,
    )

    map_processor.run()


if __name__ == '__main__':
    generic_processing_test__specified_extent()
    generic_processing_test__specified_vlayer_mask()
    print('Done')
