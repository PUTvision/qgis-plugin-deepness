import os
from glob import glob
from test.test_utils import (create_default_input_channels_mapping_for_rgba_bands, create_rlayer_from_file,
                             create_vlayer_from_file, get_dummy_fotomap_area_path, get_dummy_fotomap_small_path,
                             init_qgis)
from unittest.mock import MagicMock

import cv2
import numpy as np

from deepness.common.processing_overlap import ProcessingOverlap, ProcessingOverlapOptions
from deepness.common.processing_parameters.map_processing_parameters import ProcessedAreaType
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
    )

    map_processor = MapProcessorTrainingDataExport(
        rlayer=rlayer,
        vlayer_mask=vlayer,  # layer with masks
        map_canvas=MagicMock(),
        params=params,
    )

    map_processor.run()

    images_results = glob(os.path.join(map_processor.output_dir_path, '*_img_*.png'))
    masks_results = glob(os.path.join(map_processor.output_dir_path, '*_mask_*.png'))
    
    assert len(images_results) == 4
    assert len(masks_results) == 4
    
    mask_values = [
        (237225, 24919),
        (236341, 25803),
        (140591, 121553),
        (133202, 128942)
    ]
    
    
    for i, mask_file in enumerate(masks_results):
        mask = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)
        
        assert len(mask.shape) == 2
        assert mask.shape[0] == 512
        assert mask.shape[1] == 512
        
        assert np.unique(mask).tolist() == [0, 255]
        
        assert np.isclose(np.sum(mask < 128), mask_values[i][0], atol=10)
        assert np.isclose(np.sum(mask >= 128), mask_values[i][1], atol=10)


if __name__ == '__main__':
    # test_export_google_earth()
    test_export_dummy_fotomap()
    print('Done')
