""" This file implements core map processing logic """

import logging
from typing import List, Optional, Tuple

import numpy as np
from qgis.core import QgsRasterLayer, QgsTask, QgsVectorLayer
from qgis.gui import QgsMapCanvas
from qgis.PyQt.QtCore import pyqtSignal

from deepness.common.defines import IS_DEBUG
from deepness.common.lazy_package_loader import LazyPackageLoader
from deepness.common.processing_parameters.map_processing_parameters import MapProcessingParameters, ProcessedAreaType
from deepness.common.temp_files_handler import TempFilesHandler
from deepness.processing import extent_utils, processing_utils
from deepness.processing.map_processor.map_processing_result import MapProcessingResult, MapProcessingResultFailed
from deepness.processing.tile_params import TileParams

cv2 = LazyPackageLoader('cv2')


class MapProcessor(QgsTask):
    """
    Base class for processing the ortophoto with parameters received from the UI.

    Actual processing is done in specialized child classes. Here we have the "core" functionality,
    like iterating over single tiles.

    Objects of this class are created and managed by the 'Deepness'.
    Work is done within QgsTask, for seamless integration with QGis GUI and logic.
    """

    # error message if finished with error, empty string otherwise
    finished_signal = pyqtSignal(MapProcessingResult)
    # request to show an image. Params: (image, window_name)
    show_img_signal = pyqtSignal(object, str)

    def __init__(self,
                 rlayer: QgsRasterLayer,
                 vlayer_mask: Optional[QgsVectorLayer],
                 map_canvas: QgsMapCanvas,
                 params: MapProcessingParameters):
        """ init
        Parameters
        ----------
        rlayer : QgsRasterLayer
            Raster layer which is being processed
        vlayer_mask : Optional[QgsVectorLayer]
            Vector layer with outline of area which should be processed (within rlayer)
        map_canvas : QgsMapCanvas
            active map canvas (in the GUI), required if processing visible map area
        params : MapProcessingParameters
           see MapProcessingParameters
        """
        QgsTask.__init__(self, self.__class__.__name__)
        self._processing_finished = False
        self.rlayer = rlayer
        self.vlayer_mask = vlayer_mask
        self.params = params
        self._assert_qgis_doesnt_need_reload()
        self._processing_result = MapProcessingResultFailed('Failed to get processing result!')

        self.stride_px = self.params.processing_stride_px  # stride in pixels
        self.rlayer_units_per_pixel = processing_utils.convert_meters_to_rlayer_units(
            self.rlayer, self.params.resolution_m_per_px)  # number of rlayer units for one tile pixel

        self.file_handler = TempFilesHandler() if self.params.local_cache else None

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

        # Coordinate of base image within extended image (images for base_extent and extended_extent)
        self.base_extent_bbox_in_full_image = extent_utils.calculate_base_extent_bbox_in_full_image(
            image_size_y=self.img_size_y_pixels,
            base_extent=self.base_extent,
            extended_extent=self.extended_extent,
            rlayer_units_per_pixel=self.rlayer_units_per_pixel)

        # Number of tiles in x and y dimensions which will be used during processing
        # As we are using "extended_extent" this should divide without any rest
        self.x_bins_number = round((self.img_size_x_pixels - self.params.tile_size_px) / self.stride_px) + 1
        self.y_bins_number = round((self.img_size_y_pixels - self.params.tile_size_px) / self.stride_px) + 1

        # Mask determining area to process (within extended_extent coordinates)
        self.area_mask_img = processing_utils.create_area_mask_image(
            vlayer_mask=self.vlayer_mask,
            rlayer=self.rlayer,
            extended_extent=self.extended_extent,
            rlayer_units_per_pixel=self.rlayer_units_per_pixel,
            image_shape_yx=(self.img_size_y_pixels, self.img_size_x_pixels),
            files_handler=self.file_handler)  # type: Optional[np.ndarray]

        self._result_img = None

    def set_results_img(self, img):
        if self._result_img is not None:
            raise Exception("Result image already created!")

        self._result_img = img

    def get_result_img(self):
        if self._result_img is None:
            raise Exception("Result image not yet created!")

        return self._result_img

    def _assert_qgis_doesnt_need_reload(self):
        """ If the plugin is somehow invalid, it cannot compare the enums correctly
        I suppose it could be fixed somehow, but no need to investigate it now,
        it affects only the development
        """

        if self.params.processed_area_type.__class__ != ProcessedAreaType:
            raise Exception("Disable plugin, restart QGis and enable plugin again!")

    def run(self):
        try:
            self._processing_result = self._run()
        except Exception as e:
            logging.exception("Error occurred in MapProcessor:")
            msg = "Unhandled exception occurred. See Python Console for details"
            self._processing_result = MapProcessingResultFailed(msg, exception=e)
            if IS_DEBUG:
                raise e

        self._processing_finished = True
        return True

    def _run(self) -> MapProcessingResult:
        raise NotImplementedError('Base class not implemented!')

    def finished(self, result: bool):
        if result:
            gui_delegate = self._processing_result.gui_delegate
            if gui_delegate is not None:
                gui_delegate()
        else:
            self._processing_result = MapProcessingResultFailed("Unhandled processing error!")
        self.finished_signal.emit(self._processing_result)

    @staticmethod
    def is_busy():
        return True

    def _show_image(self, img, window_name='img'):
        self.show_img_signal.emit(img, window_name)

    def limit_extended_extent_image_to_base_extent_with_mask(self, full_img):
        """
        Limit an image which is for extended_extent to the base_extent image.
        If a limiting polygon was used for processing, it will be also applied.
        :param full_img:
        :return:
        """
        # TODO look for some inplace operation to save memory
        # cv2.copyTo(src=full_img, mask=area_mask_img, dst=full_img)  # this doesn't work due to implementation details
        
        for i in range(full_img.shape[0]):
            full_img[i] = cv2.copyTo(src=full_img[i], mask=self.area_mask_img)

        b = self.base_extent_bbox_in_full_image
        result_img = full_img[:, b.y_min:b.y_max+1, b.x_min:b.x_max+1]
        return result_img

    def _get_array_or_mmapped_array(self, final_shape_px):
        if self.file_handler is not None:
            full_result_img = np.memmap(
                self.file_handler.get_results_img_path(),
                dtype=np.uint8,
                mode='w+',
                shape=final_shape_px)
        else:
            full_result_img = np.zeros(final_shape_px, np.uint8)

        return full_result_img

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

    def tiles_generator_batched(self) -> Tuple[np.ndarray, List[TileParams]]:
        """
        Iterate over all tiles, as a Python generator function, but return them in batches
        """

        tile_img_batch, tile_params_batch = [], []

        for tile_img, tile_params in self.tiles_generator():
            tile_img_batch.append(tile_img)
            tile_params_batch.append(tile_params)

            if len(tile_img_batch) >= self.params.batch_size:
                yield np.array(tile_img_batch), tile_params_batch
                tile_img_batch, tile_params_batch = [], []

        if len(tile_img_batch) > 0:
            yield np.array(tile_img_batch), tile_params_batch
            tile_img_batch, tile_params_batch = [], []
