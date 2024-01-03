from test.test_utils import (create_default_input_channels_mapping_for_rgba_bands, create_rlayer_from_file,
                             create_vlayer_from_file, get_dummy_fotomap_area_path, get_dummy_fotomap_small_path,
                             init_qgis)
from unittest.mock import MagicMock

from deepness.common.processing_overlap import ProcessingOverlap, ProcessingOverlapOptions
from deepness.common.processing_parameters.map_processing_parameters import ModelOutputFormat, ProcessedAreaType
from deepness.common.processing_parameters.training_data_export_parameters import TrainingDataExportParameters
from deepness.processing.map_processor.map_processor_training_data_export import MapProcessorTrainingDataExport

RASTER_FILE_PATH = get_dummy_fotomap_small_path()


def test_export_dummy_fotomap():
    qgs = init_qgis()

    rlayer = create_rlayer_from_file(RASTER_FILE_PATH)
    vlayer = create_vlayer_from_file(get_dummy_fotomap_area_path())

    params = TrainingDataExportParameters(
        export_image_tiles=True,
        resolution_cm_per_px=3,
        batch_size=1,
        local_cache=False,
        segmentation_mask_layer_id=vlayer.id(),
        output_directory_path='/tmp/qgis_test',
        tile_size_px=512,  # same x and y dimensions, so take x
        processed_area_type=ProcessedAreaType.ENTIRE_LAYER,
        mask_layer_id=None,
        input_layer_id=rlayer.id(),
        input_channels_mapping=create_default_input_channels_mapping_for_rgba_bands(),
        processing_overlap=ProcessingOverlap(ProcessingOverlapOptions.OVERLAP_IN_PERCENT, percentage=20),
        model_output_format=ModelOutputFormat.ALL_CLASSES_AS_SEPARATE_LAYERS,
        model_output_format__single_class_number=-1,
    )

    map_processor = MapProcessorTrainingDataExport(
        rlayer=rlayer,
        vlayer_mask=vlayer,  # layer with masks
        map_canvas=MagicMock(),
        params=params,
    )

    map_processor.run()
    # TODO - validate processing result (we expect to have xxx tiles in directory yyy)

#
# def test_export_google_earth():
#     """
#     Just a test to debug part of processing with Google Earth Satellite images.
#     idk how to create this layer in Python, so I loaded a project which contains this layer.
#     But then, this layer works only partially, therefore this test is commented
#     :return:
#     """
#     qgs = init_qgis()
#
#     project = QgsProject.instance()
#     project.read('/home/przemek/Desktop/corn/borecko/qq.qgz')
#     for layer_id, layer in project.mapLayers().items():
#         if 'Google Satellite' in layer.name():
#             rlayer = layer
#             break
#
#     if not rlayer.dataProvider():
#         # it looks like the google satellite layer is not working outside of GUI,
#         # even if loading from project where it is
#         print('Cannot perform "export_google_earth_test" - cannot use google satellite layer')
#         return
#
#     params = TrainingDataExportParameters(
#         export_image_tiles=True,
#         resolution_cm_per_px=3,
#         segmentation_mask_layer_id=None,
#         output_directory_path='/tmp/qgis_test',
#         tile_size_px=512,  # same x and y dimensions, so take x
#         processed_area_type=ProcessedAreaType.VISIBLE_PART,
#         mask_layer_id=None,
#         input_layer_id=rlayer.id(),
#         input_channels_mapping=create_default_input_channels_mapping_for_google_satellite_bands(),
#         processing_overlap=20,
#     )
#
#     processed_extent = QgsRectangle(
#         1881649.80, 6867603.86,
#         1881763.08, 6867662.50)
#
#     map_canvas = MagicMock()
#     map_canvas.extent = lambda: processed_extent
#     map_canvas.mapSettings().destinationCrs = lambda: QgsCoordinateReferenceSystem("EPSG:3857")
#
#     map_processor = MapProcessorTrainingDataExport(
#         rlayer=rlayer,
#         vlayer_mask=None,
#         map_canvas=map_canvas,
#         params=params,
#     )
#
#     map_processor.run()


if __name__ == '__main__':
    # test_export_google_earth()
    test_export_dummy_fotomap()
    print('Done')
