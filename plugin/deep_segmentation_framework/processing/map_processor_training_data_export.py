import datetime
from typing import Optional, Tuple
import os

import numpy as np
import cv2

from qgis.PyQt.QtCore import pyqtSignal
from qgis.core import QgsVectorLayer
from qgis.gui import QgsMapCanvas
from qgis.core import QgsRasterLayer
from qgis.core import QgsTask
from qgis.core import QgsProject

from deep_segmentation_framework.common.processing_parameters.map_processing_parameters import MapProcessingParameters
from deep_segmentation_framework.common.processing_parameters.training_data_export_parameters import \
    TrainingDataExportParameters
from deep_segmentation_framework.processing import processing_utils, extent_utils
from deep_segmentation_framework.common.defines import IS_DEBUG
from deep_segmentation_framework.common.processing_parameters.inference_parameters import InferenceParameters
from deep_segmentation_framework.processing.map_processor import MapProcessor
from deep_segmentation_framework.processing.tile_params import TileParams

if IS_DEBUG:
    from matplotlib import pyplot as plt


class MapProcessorTrainingDataExport(MapProcessor):
    def __init__(self,
                 params: TrainingDataExportParameters,
                 **kwargs):
        super().__init__(
            params=params,
            **kwargs)
        self.params = params
        self.output_dir_path = self._create_output_dir()

    def _create_output_dir(self) -> str:
        datetime_string = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
        full_path = os.path.join(self.params.output_directory_path, datetime_string)
        os.makedirs(full_path, exist_ok=True)
        return full_path

    def _run(self):
        for tile_img, tile_params in self.tiles_generator():
            if self.isCanceled():
                return False
            tile_params = tile_params  # type: TileParams
            file_name = f'tile_img_{tile_params.x_bin_number}_{tile_params.y_bin_number}.png'
            file_path = os.path.join(self.output_dir_path, file_name)

            if tile_img.shape[-1] == 4:
                tile_img = cv2.cvtColor(tile_img, cv2.COLOR_RGBA2BGRA)

            cv2.imwrite(file_path, tile_img)

            # TODO: save mask
        return True
