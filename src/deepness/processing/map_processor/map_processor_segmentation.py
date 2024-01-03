""" This file implements map processing for segmentation model """

import numpy as np
from qgis.core import QgsProject, QgsVectorLayer

from deepness.common.lazy_package_loader import LazyPackageLoader
from deepness.common.processing_parameters.segmentation_parameters import SegmentationParameters
from deepness.processing import processing_utils
from deepness.processing.map_processor.map_processing_result import (MapProcessingResult, MapProcessingResultCanceled,
                                                                     MapProcessingResultSuccess)
from deepness.processing.map_processor.map_processor_with_model import MapProcessorWithModel

cv2 = LazyPackageLoader('cv2')


class MapProcessorSegmentation(MapProcessorWithModel):
    """
    MapProcessor specialized for Segmentation model (where each pixel is assigned to one class).

    Implementation note: due to opencv operations on arrays, it is easier to use value 0 for special meaning,
    (that is pixel out of processing area) instead of just a class with number 0.
    Therefore, internally during processing, pixels representing classes have value `class_number + 1`.
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
        
        if self.file_handler is not None:
            full_result_img = np.memmap(
                self.file_handler.get_results_img_path(),
                dtype=np.uint8,
                mode='w+',
                shape=final_shape_px)
        else:
            full_result_img = np.zeros(final_shape_px, np.uint8)
            
        for tile_img, tile_params in self.tiles_generator():
            if self.isCanceled():
                return MapProcessingResultCanceled()

            # See note in the class description why are we adding/subtracting 1 here
            tile_result = self._process_tile(tile_img) + 1

            tile_params.set_mask_on_full_img(
                tile_result=tile_result,
                full_result_img=full_result_img)

        blur_size = int(self.segmentation_parameters.postprocessing_dilate_erode_size // 2) * 2 + 1  # needs to be odd
        full_result_img = cv2.medianBlur(full_result_img, blur_size)
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

        # we cannot simply take image dimensions, because we may have irregular processing area from polygon
        number_of_pixels_in_processing_area = np.sum([counts_map[k] for k in counts_map.keys() if k != 0])
        total_area = number_of_pixels_in_processing_area * self.params.resolution_m_per_px**2
        for channel_id in channels:
            # See note in the class description why are we adding/subtracting 1 here
            pixels_count = counts_map.get(channel_id + 1, 0)
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
            # See note in the class description why are we adding/subtracting 1 here
            local_mask_img = np.uint8(mask_img == (channel_id + 1))

            # remove small areas - old implementation. Now we decided to do median blur, because the method below
            # was producing pixels not belonging to any class
            # local_mask_img = processing_utils.erode_dilate_image(
            #     img=local_mask_img,
            #     segmentation_parameters=self.segmentation_parameters)

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
        if (result.shape[0] == 1):
            result = (result != 0).astype(int)[0]         
        else:
            result = np.argmax(result, axis=0)
        return result
