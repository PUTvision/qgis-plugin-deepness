""" This file implements map processing for segmentation model """

import cv2
import numpy as np
from qgis.core import QgsProject
from qgis.core import QgsVectorLayer

from deepness.common.processing_parameters.segmentation_parameters import SegmentationParameters
from deepness.processing import processing_utils
from deepness.processing.map_processor.map_processing_result import MapProcessingResult, \
    MapProcessingResultCanceled, MapProcessingResultSuccess
from deepness.processing.map_processor.map_processor_with_model import MapProcessorWithModel


class MapProcessorSegmentation(MapProcessorWithModel):
    """
    MapProcessor specialized for Segmentation model (where each pixel is assigned to one class)
    """

    def __init__(self,
                 params: SegmentationParameters,
                 **kwargs):
        super().__init__(
            params=params,
            model=params.model,
            **kwargs)
        self.segmentation_parameters = params
        self.model = params.model
        self._result_img = None

    def get_result_img(self):
        return self._result_img

    def _run(self) -> MapProcessingResult:
        final_shape_px = (self.img_size_y_pixels, self.img_size_x_pixels)
        full_result_img = np.zeros(final_shape_px, np.uint8)
        for tile_img, tile_params in self.tiles_generator():
            if self.isCanceled():
                return MapProcessingResultCanceled()

            tile_result = self._process_tile(tile_img)
            # self._show_image(tile_result)
            tile_params.set_mask_on_full_img(
                tile_result=tile_result,
                full_result_img=full_result_img)

        full_result_img = processing_utils.erode_dilate_image(
            img=full_result_img,
            segmentation_parameters=self.segmentation_parameters)
        # plt.figure(); plt.imshow(full_result_img); plt.show(block=False); plt.pause(0.001)
        self._result_img = self.limit_extended_extent_image_to_base_extent_with_mask(full_img=full_result_img)
        self._create_vlayer_from_mask_for_base_extent(self._result_img)

        result_message = self._create_result_message(self._result_img)
        return MapProcessingResultSuccess(result_message)

    def _create_result_message(self, result_img: np.ndarray) -> str:
        unique, counts = np.unique(result_img, return_counts=True)
        counts_map = {}
        for i in range(len(unique)):
            counts_map[unique[i]] = counts[i]

        channels = self._get_indexes_of_model_output_channels_to_create()
        txt = f'Segmentation done for {len(channels)} model output channels, with the following statistics:\n'
        total_area = result_img.shape[0] * result_img.shape[1] * self.params.resolution_m_per_px**2
        for channel_id in channels:
            pixels_count = counts_map.get(channel_id, 0)
            area = pixels_count * self.params.resolution_m_per_px**2
            if total_area:
                area_percentage = area / total_area * 100
            else:
                area_percentage = 0.0
            txt += f' - {self.model.get_channel_name(channel_id)}: area = {area:.2f} m^2 ({area_percentage:.2f} %)\n'

        return txt

    def _create_vlayer_from_mask_for_base_extent(self, mask_img):
        # create vector layer with polygons from the mask image

        group = QgsProject.instance().layerTreeRoot().insertGroup(0, 'model_output')

        for channel_id in self._get_indexes_of_model_output_channels_to_create():
            local_mask_img = np.uint8(mask_img == channel_id)

            contours, hierarchy = cv2.findContours(local_mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

            vlayer = QgsVectorLayer("multipolygon", self.model.get_channel_name(channel_id), "memory")
            vlayer.setCrs(self.rlayer.crs())
            prov = vlayer.dataProvider()

            color = vlayer.renderer().symbol().color()
            OUTPUT_VLAYER_COLOR_TRANSPARENCY = 80
            color.setAlpha(OUTPUT_VLAYER_COLOR_TRANSPARENCY)
            vlayer.renderer().symbol().setColor(color)
            # TODO - add also outline for the layer (thicker black border)

            prov.addFeatures(features)
            vlayer.updateExtents()

            QgsProject.instance().addMapLayer(vlayer, False)
            group.addLayer(vlayer)

    def _process_tile(self, tile_img: np.ndarray) -> np.ndarray:
        # TODO - create proper mapping for output channels
        result = self.model.process(tile_img)

        result[result < self.segmentation_parameters.pixel_classification__probability_threshold] = 0.0
        result = np.argmax(result, axis=0)
        return result
