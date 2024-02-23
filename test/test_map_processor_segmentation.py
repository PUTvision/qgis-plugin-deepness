from test.test_utils import (create_default_input_channels_mapping_for_rgba_bands, create_rlayer_from_file,
                             create_vlayer_from_file, get_dummy_fotomap_area_crs3857_path, get_dummy_fotomap_area_path,
                             get_dummy_fotomap_small_path, get_dummy_segmentation_model_path, init_qgis)
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
from qgis.core import QgsCoordinateReferenceSystem, QgsRectangle

from deepness.common.processing_overlap import ProcessingOverlap, ProcessingOverlapOptions
from deepness.common.processing_parameters.map_processing_parameters import ProcessedAreaType
from deepness.common.processing_parameters.segmentation_parameters import SegmentationParameters
from deepness.processing.map_processor.map_processor_segmentation import MapProcessorSegmentation
from deepness.processing.models.segmentor import Segmentor

RASTER_FILE_PATH = get_dummy_fotomap_small_path()

VLAYER_MASK_FILE_PATH = get_dummy_fotomap_area_path()

VLAYER_MASK_CRS3857_FILE_PATH = get_dummy_fotomap_area_crs3857_path()

MODEL_FILE_PATH = get_dummy_segmentation_model_path()

INPUT_CHANNELS_MAPPING = create_default_input_channels_mapping_for_rgba_bands()

PROCESSED_EXTENT_1 = QgsRectangle(  # big part of the fotomap
    638840.370, 5802593.197,
    638857.695, 5802601.792)

def model_process_mock_one_channel(x):
    x = x[:, :, :, 0:1]
    x = np.transpose(x, (0, 3, 1, 2))

    return x

def model_process_mock_two_channels(x):    
    return [model_process_mock_one_channel(x), model_process_mock_one_channel(x)]


def test_dummy_model_processing__entire_file():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    model = Segmentor(MODEL_FILE_PATH)

    params = SegmentationParameters(
        resolution_cm_per_px=3,
        tile_size_px=model.get_input_size_in_pixels()[0],  # same x and y dimensions, so take x
        batch_size=1,
        local_cache=False,
        processed_area_type=ProcessedAreaType.ENTIRE_LAYER,
        mask_layer_id=None,
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        postprocessing_dilate_erode_size=5,
        processing_overlap=ProcessingOverlap(ProcessingOverlapOptions.OVERLAP_IN_PERCENT, percentage=20),
        pixel_classification__probability_threshold=0.5,
        model=model,
    )

    map_processor = MapProcessorSegmentation(
        rlayer=rlayer,
        vlayer_mask=None,
        map_canvas=MagicMock(),
        params=params,
    )

    map_processor.run()
    result_img = map_processor.get_result_img()

    assert result_img.shape == (1, 561, 829)
    # TODO - add detailed check for pixel values once we have output channels mapping with thresholding

def test_dummy_model_processing__entire_file_overlap_in_pixels():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    model = Segmentor(MODEL_FILE_PATH)

    params = SegmentationParameters(
        resolution_cm_per_px=3,
        tile_size_px=model.get_input_size_in_pixels()[0],  # same x and y dimensions, so take x
        batch_size=1,
        local_cache=False,
        processed_area_type=ProcessedAreaType.ENTIRE_LAYER,
        mask_layer_id=None,
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        postprocessing_dilate_erode_size=5,
        processing_overlap=ProcessingOverlap(ProcessingOverlapOptions.OVERLAP_IN_PIXELS, overlap_px=int(model.get_input_size_in_pixels()[0] * 0.2)),
        pixel_classification__probability_threshold=0.5,
        model=model,
    )

    map_processor = MapProcessorSegmentation(
        rlayer=rlayer,
        vlayer_mask=None,
        map_canvas=MagicMock(),
        params=params,
    )

    map_processor.run()
    result_img = map_processor.get_result_img()

    assert result_img.shape == (1, 561, 829)


def test_generic_processing_test__specified_extent_from_vlayer_one_channel():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    vlayer_mask = create_vlayer_from_file(VLAYER_MASK_FILE_PATH)
    model = MagicMock()
    model.process = model_process_mock_one_channel
    model.get_number_of_channels = lambda: 3
    model.get_number_of_output_channels = lambda: [1]
    model.get_channel_name = lambda y, x: str(y)+'_'+str(x)

    params = SegmentationParameters(
        resolution_cm_per_px=3,
        tile_size_px=512,
        batch_size=1,
        local_cache=False,
        processed_area_type=ProcessedAreaType.FROM_POLYGONS,
        mask_layer_id=vlayer_mask.id(),
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        postprocessing_dilate_erode_size=5,
        processing_overlap=ProcessingOverlap(ProcessingOverlapOptions.OVERLAP_IN_PERCENT, percentage=20),
        pixel_classification__probability_threshold=0.5,
        model=model,
    )
    map_processor = MapProcessorSegmentation(
        rlayer=rlayer,
        vlayer_mask=vlayer_mask,
        map_canvas=MagicMock(),
        params=params,
    )

    # just run - we will check the results in a more detailed test
    map_processor.run()
    result_img = map_processor.get_result_img()
    assert result_img.shape == (1, 524, 733)

    # just check a few pixels
    assert all(result_img.ravel()[[365, 41234, 59876, 234353, 111222, 134534, 223423, 65463, 156451]] ==
               np.asarray([0, 1, 1, 1, 1, 0, 0, 1, 0]))
    
    # and counts of different values
    np.testing.assert_allclose(np.unique(result_img, return_counts=True)[1], np.array([178063, 206029]), atol=3)

def test_generic_processing_test__specified_extent_from_vlayer_two_channels():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    vlayer_mask = create_vlayer_from_file(VLAYER_MASK_FILE_PATH)
    model = MagicMock()
    model.process = model_process_mock_two_channels
    model.get_number_of_channels = lambda: 3
    model.get_number_of_output_channels = lambda: [1, 1]
    model.get_channel_name = lambda y, x: str(y)+'_'+str(x)

    params = SegmentationParameters(
        resolution_cm_per_px=3,
        tile_size_px=512,
        batch_size=1,
        local_cache=False,
        processed_area_type=ProcessedAreaType.FROM_POLYGONS,
        mask_layer_id=vlayer_mask.id(),
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        postprocessing_dilate_erode_size=5,
        processing_overlap=ProcessingOverlap(ProcessingOverlapOptions.OVERLAP_IN_PERCENT, percentage=20),
        pixel_classification__probability_threshold=0.5,
        model=model,
    )
    map_processor = MapProcessorSegmentation(
        rlayer=rlayer,
        vlayer_mask=vlayer_mask,
        map_canvas=MagicMock(),
        params=params,
    )

    # just run - we will check the results in a more detailed test
    map_processor.run()
    result_img = map_processor.get_result_img()
    assert result_img.shape == (2, 524, 733)

    # just check a few pixels
    assert all(result_img.ravel()[[365, 41234, 59876, 234353, 111222, 134534, 223423, 65463, 156451]] ==
               np.asarray([0, 1, 1, 1, 1, 0, 0, 1, 0]))
    
    # and counts of different values
    np.testing.assert_allclose(np.unique(result_img, return_counts=True)[1], np.array([356126, 412058]), atol=3)


def test_generic_processing_test__specified_extent_from_vlayer_crs3857_one_channel():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    vlayer_mask = create_vlayer_from_file(VLAYER_MASK_CRS3857_FILE_PATH)
    model = MagicMock()
    model.process = model_process_mock_one_channel
    model.get_number_of_channels = lambda: 3
    model.get_number_of_output_channels = lambda: [1]
    model.get_channel_name = lambda y, x: str(y)+'_'+str(x)

    params = SegmentationParameters(
        resolution_cm_per_px=3,
        tile_size_px=512,
        batch_size=1,
        local_cache=True,
        processed_area_type=ProcessedAreaType.FROM_POLYGONS,
        mask_layer_id=vlayer_mask.id(),
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        postprocessing_dilate_erode_size=5,
        processing_overlap=ProcessingOverlap(ProcessingOverlapOptions.OVERLAP_IN_PERCENT, percentage=20),
        pixel_classification__probability_threshold=0.5,
        model=model,
    )
    map_processor = MapProcessorSegmentation(
        rlayer=rlayer,
        vlayer_mask=vlayer_mask,
        map_canvas=MagicMock(),
        params=params,
    )

    # just run - we will check the results in a more detailed test
    map_processor.run()
    result_img = map_processor.get_result_img()
    assert result_img.shape == (1, 550, 723)

    # just check a few pixels
    assert all(result_img.ravel()[[365, 41234, 59876, 234353, 111222, 134534, 223423, 65463, 156451]] ==
               np.asarray([0, 0, 1, 1, 1, 0, 0, 1, 0]))
    # and counts of different values
    np.testing.assert_allclose(np.unique(result_img, return_counts=True)[1], np.array([193853, 203797]), atol=3)

def test_generic_processing_test__specified_extent_from_vlayer_crs3857_two_channels():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    vlayer_mask = create_vlayer_from_file(VLAYER_MASK_CRS3857_FILE_PATH)
    model = MagicMock()
    model.process = model_process_mock_two_channels
    model.get_number_of_channels = lambda: 3
    model.get_number_of_output_channels = lambda: [1, 1]
    model.get_channel_name = lambda y, x: str(y)+'_'+str(x)

    params = SegmentationParameters(
        resolution_cm_per_px=3,
        tile_size_px=512,
        batch_size=1,
        local_cache=False,
        processed_area_type=ProcessedAreaType.FROM_POLYGONS,
        mask_layer_id=vlayer_mask.id(),
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        postprocessing_dilate_erode_size=5,
        processing_overlap=ProcessingOverlap(ProcessingOverlapOptions.OVERLAP_IN_PERCENT, percentage=20),
        pixel_classification__probability_threshold=0.5,
        model=model,
    )
    map_processor = MapProcessorSegmentation(
        rlayer=rlayer,
        vlayer_mask=vlayer_mask,
        map_canvas=MagicMock(),
        params=params,
    )

    # just run - we will check the results in a more detailed test
    map_processor.run()
    result_img = map_processor.get_result_img()

    # for the same vlayer_mask, but with a different encoding we had quite different shaep (524, 733).
    # I'm not sure if it is rounding issue in Qgis Transform or some bug in plugin
    assert result_img.shape == (2, 550, 723)

    # just check a few pixels
    assert all(result_img.ravel()[[365, 41234, 59876, 234353, 111222, 134534, 223423, 65463, 156451]] ==
               np.asarray([0, 0, 1, 1, 1, 0, 0, 1, 0]))
    
    np.testing.assert_allclose(np.unique(result_img, return_counts=True)[1], np.array([387706, 407594]), atol=3)


def test_generic_processing_test__specified_extent_from_active_map_extent_one_channel():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    model = MagicMock()
    model.process = model_process_mock_one_channel
    model.get_number_of_channels = lambda: 3
    model.get_number_of_output_channels = lambda: [1]
    model.get_channel_name = lambda y, x: str(y)+'_'+str(x)

    params = SegmentationParameters(
        resolution_cm_per_px=3,
        tile_size_px=512,
        batch_size=1,
        local_cache=False,
        processed_area_type=ProcessedAreaType.VISIBLE_PART,
        mask_layer_id=None,
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        postprocessing_dilate_erode_size=5,
        processing_overlap=ProcessingOverlap(ProcessingOverlapOptions.OVERLAP_IN_PERCENT, percentage=20),
        pixel_classification__probability_threshold=0.5,
        model=model,
    )
    processed_extent = PROCESSED_EXTENT_1

    # we want to use a fake extent, which is the Visible Part of the map,
    # so we need to mock its function calls
    params.processed_area_type = ProcessedAreaType.VISIBLE_PART
    map_canvas = MagicMock()
    map_canvas.extent = lambda: processed_extent
    map_canvas.mapSettings().destinationCrs = lambda: QgsCoordinateReferenceSystem("EPSG:32633")

    map_processor = MapProcessorSegmentation(
        rlayer=rlayer,
        vlayer_mask=None,
        map_canvas=map_canvas,
        params=params,
    )

    # just run - we will check the results in a more detailed test
    map_processor.run()

def test_generic_processing_test__specified_extent_from_active_map_extent_two_channels():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    model = MagicMock()
    model.process = model_process_mock_two_channels
    model.get_number_of_channels = lambda: 3
    model.get_number_of_output_channels = lambda: [1, 1]
    model.get_channel_name = lambda y, x: str(y)+'_'+str(x)

    params = SegmentationParameters(
        resolution_cm_per_px=3,
        tile_size_px=512,
        batch_size=1,
        local_cache=False,
        processed_area_type=ProcessedAreaType.VISIBLE_PART,
        mask_layer_id=None,
        input_layer_id=rlayer.id(),
        input_channels_mapping=INPUT_CHANNELS_MAPPING,
        postprocessing_dilate_erode_size=5,
        processing_overlap=ProcessingOverlap(ProcessingOverlapOptions.OVERLAP_IN_PERCENT, percentage=20),
        pixel_classification__probability_threshold=0.5,
        model=model,
    )
    processed_extent = PROCESSED_EXTENT_1

    # we want to use a fake extent, which is the Visible Part of the map,
    # so we need to mock its function calls
    params.processed_area_type = ProcessedAreaType.VISIBLE_PART
    map_canvas = MagicMock()
    map_canvas.extent = lambda: processed_extent
    map_canvas.mapSettings().destinationCrs = lambda: QgsCoordinateReferenceSystem("EPSG:32633")

    map_processor = MapProcessorSegmentation(
        rlayer=rlayer,
        vlayer_mask=None,
        map_canvas=map_canvas,
        params=params,
    )

    # just run - we will check the results in a more detailed test
    map_processor.run()

if __name__ == '__main__':
    # test_dummy_model_processing__entire_file()
    test_generic_processing_test__specified_extent_from_vlayer_one_channel()
    test_generic_processing_test__specified_extent_from_vlayer_two_channels()
    test_generic_processing_test__specified_extent_from_vlayer_crs3857_one_channel()
    test_generic_processing_test__specified_extent_from_vlayer_crs3857_two_channels()
    test_generic_processing_test__specified_extent_from_active_map_extent_one_channel()
    test_generic_processing_test__specified_extent_from_active_map_extent_two_channels()
    print('Done')
