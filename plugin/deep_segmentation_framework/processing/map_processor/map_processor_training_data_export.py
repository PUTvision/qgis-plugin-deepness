import datetime
import os

import cv2

from qgis.core import QgsProject

from deep_segmentation_framework.common.processing_parameters.training_data_export_parameters import \
    TrainingDataExportParameters
from deep_segmentation_framework.processing import processing_utils
from deep_segmentation_framework.processing.map_processor.map_processing_result import MapProcessingResultSuccess, \
    MapProcessingResultCanceled
from deep_segmentation_framework.processing.map_processor.map_processor import MapProcessor
from deep_segmentation_framework.processing.tile_params import TileParams


class MapProcessorTrainingDataExport(MapProcessor):
    """
    Map Processor specialized in exporting training data, not doing any prediction with model.
    Exports tiles for the ortophoto and a mask layer.
    """

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

        export_segmentation_mask = self.params.segmentation_mask_layer_id is not None
        if export_segmentation_mask:
            vlayer_segmentation = QgsProject.instance().mapLayers()[self.params.segmentation_mask_layer_id]
            vlayer_segmentation.setCrs(self.rlayer.crs())
            segmentation_mask_full = processing_utils.create_area_mask_image(
                vlayer_mask=vlayer_segmentation,
                extended_extent=self.extended_extent,
                rlayer_units_per_pixel=self.rlayer_units_per_pixel,
                image_shape_yx=[self.img_size_y_pixels, self.img_size_x_pixels])

        number_of_written_tiles = 0
        for tile_img, tile_params in self.tiles_generator():
            if self.isCanceled():
                return MapProcessingResultCanceled()

            tile_params = tile_params  # type: TileParams

            if self.params.export_image_tiles:
                file_name = f'tile_img_{tile_params.x_bin_number}_{tile_params.y_bin_number}.png'
                file_path = os.path.join(self.output_dir_path, file_name)
                if tile_img.shape[-1] == 4:
                    tile_img = cv2.cvtColor(tile_img, cv2.COLOR_RGBA2BGRA)
                elif tile_img.shape[-1] == 3:
                    tile_img = cv2.cvtColor(tile_img, cv2.COLOR_RGB2BGR)

                cv2.imwrite(file_path, tile_img)
                number_of_written_tiles += 1

            if export_segmentation_mask:
                file_name = f'tile_mask_{tile_params.x_bin_number}_{tile_params.y_bin_number}.png'
                file_path = os.path.join(self.output_dir_path, file_name)
                segmentation_mask_for_tile = tile_params.get_entire_tile_from_full_img(segmentation_mask_full)
                cv2.imwrite(file_path, segmentation_mask_for_tile)

        result_message = self._create_result_message(number_of_written_tiles)
        return MapProcessingResultSuccess(result_message)

    def _create_result_message(self, number_of_written_tiles) -> str:
        total_area = self.img_size_x_pixels * self.img_size_y_pixels * self.params.resolution_m_per_px**2
        return f'Exporting data finished!\n' \
               f'Exported {number_of_written_tiles} tiles.\n' \
               f'Total processed area: {total_area:.2f} m^2\n' \
               f'Directory: "{self.output_dir_path}"'
