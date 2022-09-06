from typing import Optional, Tuple

import numpy as np
import cv2

from qgis.PyQt.QtCore import pyqtSignal
from qgis.core import QgsVectorLayer
from qgis.gui import QgsMapCanvas
from qgis.core import QgsRasterLayer
from qgis.core import QgsTask
from qgis.core import QgsProject

from deep_segmentation_framework.common.processing_parameters.map_processing_parameters import MapProcessingParameters
from deep_segmentation_framework.processing import processing_utils, extent_utils
from deep_segmentation_framework.common.defines import IS_DEBUG
from deep_segmentation_framework.common.processing_parameters.inference_parameters import InferenceParameters
from deep_segmentation_framework.processing.tile_params import TileParams

if IS_DEBUG:
    from matplotlib import pyplot as plt


class MapProcessor(QgsTask):
    finished_signal = pyqtSignal(str)  # error message if finished with error, empty string otherwise
    show_img_signal = pyqtSignal(object, str)  # request to show an image. Params: (image, window_name)

    def __init__(self,
                 rlayer: QgsRasterLayer,
                 vlayer_mask: Optional[QgsVectorLayer],
                 map_canvas: QgsMapCanvas,
                 params: MapProcessingParameters):
        """

        :param rlayer: Raster layer which is being processed
        :param vlayer_mask: Vector layer with outline of area which should be processed (within rlayer)
        :param map_canvas: active map canvas (in the GUI), required if processing visible map area
        :param params: see MapProcessingParameters
        """
        QgsTask.__init__(self, self.__class__.__name__)
        self._processing_finished = False
        self.rlayer = rlayer
        self.vlayer_mask = vlayer_mask
        if vlayer_mask:
            assert vlayer_mask.crs() == self.rlayer.crs()  # should be set in higher layer
        self.params = params

        self.stride_px = self.params.processing_stride_px  # stride in pixels
        self.rlayer_units_per_pixel = processing_utils.convert_meters_to_rlayer_units(
            self.rlayer, self.params.resolution_m_per_px)  # number of rlayer units for one tile pixel

        # extent in which the actual required area is contained, without additional extensions, rounded to rlayer grid
        self.base_extent = extent_utils.calculate_base_processing_extent_in_rlayer_crs(
            map_canvas=map_canvas,
            rlayer=self.rlayer,
            vlayer_mask=self.vlayer_mask,
            params=self.params)

        # extent which should be used during model inference, as it includes extra margins to have full tiles,
        # rounded to rlayer grid
        self.extended_extent = extent_utils.calculate_extended_processing_extent(
            base_extent=self.base_extent,
            rlayer=self.rlayer,
            params=self.params,
            rlayer_units_per_pixel=self.rlayer_units_per_pixel)

        # processed rlayer dimensions (for extended_extent)
        self.img_size_x_pixels = round(self.extended_extent.width() / self.rlayer_units_per_pixel)  # how many columns (x)
        self.img_size_y_pixels = round(self.extended_extent.height() / self.rlayer_units_per_pixel)  # how many rows (y)

        # Coordinate of base image withing extended image (images for base_extent and extended_extent)
        self.base_extent_bbox_in_full_image = extent_utils.calculate_base_extent_bbox_in_full_image(
            image_size_y=self.img_size_y_pixels,
            base_extent=self.base_extent,
            extended_extent=self.extended_extent,
            rlayer_units_per_pixel=self.rlayer_units_per_pixel)

        # Number of tiles in x and y dimensions which will be used during processing
        # As we are using "extended_extent" this should divide without any rest
        self.x_bins_number = round((self.img_size_x_pixels - self.params.tile_size_px)
                                   / self.stride_px) + 1
        self.y_bins_number = round((self.img_size_y_pixels - self.params.tile_size_px)
                                   / self.stride_px) + 1

        # Mask determining area to process (within extended_extent coordinates)
        self.area_mask_img = processing_utils.create_area_mask_image(
            vlayer_mask=self.vlayer_mask,
            extended_extent=self.extended_extent,
            rlayer_units_per_pixel=self.rlayer_units_per_pixel,
            image_shape_yx=[self.img_size_y_pixels, self.img_size_x_pixels])

    def run(self):
        print('run...')
        result = self._run()
        self._processing_finished = True
        return result

    def _run(self):
        return NotImplementedError

    def finished(self, result):
        print(f'finished. Res: {result = }')
        if result:
            self.finished_signal.emit('')
        else:
            self.finished_signal.emit('Processing error')

    @staticmethod
    def is_busy():
        return True

    def _show_image(self, img, window_name='img'):
        self.show_img_signal.emit(img, window_name)

    def tiles_generator(self) -> Tuple[np.ndarray, TileParams]:
        """
        Iterate over all tiles, as a Python generator function
        """
        total_tiles = self.x_bins_number * self.y_bins_number

        for y_bin_number in range(self.y_bins_number):
            for x_bin_number in range(self.x_bins_number):
                tile_no = y_bin_number * self.x_bins_number + x_bin_number
                progress = tile_no / total_tiles * 100
                self.setProgress(progress)
                print(f" Processing tile {tile_no} / {total_tiles} [{progress:.2f}%]")
                tile_params = TileParams(
                    x_bin_number=x_bin_number, y_bin_number=y_bin_number,
                    x_bins_number=self.x_bins_number, y_bins_number=self.y_bins_number,
                    params=self.params,
                    processing_extent=self.extended_extent,
                    rlayer_units_per_pixel=self.rlayer_units_per_pixel)

                if not tile_params.is_tile_within_mask(self.area_mask_img):
                    continue  # tile outside of mask - to be skipped

                tile_img = processing_utils.get_tile_image(
                    rlayer=self.rlayer, extent=tile_params.extent, params=self.params)
                yield tile_img, tile_params
