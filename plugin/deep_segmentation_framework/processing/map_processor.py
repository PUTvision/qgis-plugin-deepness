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
        self._result_img = None
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


class MapProcessorInference(MapProcessor):
    def __init__(self,
                 inference_parameters: InferenceParameters,
                 **kwargs):
        super().__init__(
            params=inference_parameters,
            **kwargs)
        self.inference_parameters = inference_parameters
        self.model_wrapper = inference_parameters.model

    def get_result_img(self):
        return self._result_img

    def _run(self):
        final_shape_px = (self.img_size_y_pixels, self.img_size_x_pixels)
        full_result_img = np.zeros(final_shape_px, np.uint8)

        for tile_img, tile_params in self.tiles_generator():
            if self.isCanceled():
                return False

            tile_result = self._process_tile(tile_img)
            # plt.figure(); plt.imshow(tile_img); plt.show(block=False); plt.pause(0.001)
            # self._show_image(tile_result)
            tile_params.set_mask_on_full_img(
                tile_result=tile_result,
                full_result_img=full_result_img)

        full_result_img = processing_utils.erode_dilate_image(
            img=full_result_img,
            inference_parameters=self.inference_parameters)
        # plt.figure(); plt.imshow(full_result_img); plt.show(block=False); plt.pause(0.001)
        self._result_img = self.limit_extended_extent_image_to_base_extent_with_mask(full_img=full_result_img)
        self._create_vlayer_from_mask_for_base_extent(self._result_img)
        return True

    def limit_extended_extent_image_to_base_extent_with_mask(self, full_img):
        """
        Limit an image which is for extended_extent to the base_extent image.
        If a limiting polygon was used for processing, it will be also applied.
        :param full_img:
        :return:
        """
        # TODO look for some inplace operation to save memory
        # cv2.copyTo(src=full_img, mask=area_mask_img, dst=full_img)  # this doesn't work due to implementation details
        full_img = cv2.copyTo(src=full_img, mask=self.area_mask_img)

        b = self.base_extent_bbox_in_full_image
        result_img = full_img[b.y_min:b.y_max+1, b.x_min:b.x_max+1]
        return result_img

    def _create_vlayer_from_mask_for_base_extent(self, mask_img):
        # create vector layer with polygons from the mask image
        contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = processing_utils.transform_contours_yx_pixels_to_target_crs(
            contours=contours,
            extent=self.base_extent,
            rlayer_units_per_pixel=self.rlayer_units_per_pixel)
        features = []

        if len(contours):
            processing_utils.convert_cv_contours_to_features(
                features=features,
                cv_contours=contours,
                hierarchy=hierarchy[0],
                is_hole=False,
                current_holes=[],
                current_contour_index=0)
        else:
            pass  # just nothing, we already have an empty list of features

        vlayer = QgsVectorLayer("multipolygon", "model_output", "memory")
        vlayer.setCrs(self.rlayer.crs())
        prov = vlayer.dataProvider()

        color = vlayer.renderer().symbol().color()
        OUTPUT_VLAYER_COLOR_TRANSPARENCY = 80
        color.setAlpha(OUTPUT_VLAYER_COLOR_TRANSPARENCY)
        vlayer.renderer().symbol().setColor(color)
        # TODO - add also outline for the layer (thicker black border)

        prov.addFeatures(features)
        vlayer.updateExtents()
        QgsProject.instance().addMapLayer(vlayer)

    def _process_tile(self, tile_img: np.ndarray) -> np.ndarray:
        # TODO - create proper mapping for output channels
        result = self.model_wrapper.process(tile_img)

        # TODO - apply argmax classification and thresholding
        result_threshold = result > (self.inference_parameters.pixel_classification__probability_threshold * 255)

        return result_threshold

