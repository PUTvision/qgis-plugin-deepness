""" This file implements map processing for segmentation model """

from typing import Callable

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

    def _run(self) -> MapProcessingResult:
        final_shape_px = (len(self._get_indexes_of_model_output_channels_to_create()), self.img_size_y_pixels, self.img_size_x_pixels)

        full_result_img = self._get_array_or_mmapped_array(final_shape_px)

        for tile_img_batched, tile_params_batched in self.tiles_generator_batched():
            if self.isCanceled():
                return MapProcessingResultCanceled()

            tile_result_batched = self._process_tile(tile_img_batched)

            for tile_result, tile_params in zip(tile_result_batched, tile_params_batched):
                tile_params.set_mask_on_full_img(
                    tile_result=tile_result,
                    full_result_img=full_result_img)

        blur_size = int(self.segmentation_parameters.postprocessing_dilate_erode_size // 2) * 2 + 1  # needs to be odd
        
        for i in range(full_result_img.shape[0]):
            full_result_img[i] = cv2.medianBlur(full_result_img[i], blur_size)
        
        full_result_img = self.limit_extended_extent_image_to_base_extent_with_mask(full_img=full_result_img)

        self.set_results_img(full_result_img)

        gui_delegate = self._create_vlayer_from_mask_for_base_extent(self.get_result_img())

        result_message = self._create_result_message(self.get_result_img())
        return MapProcessingResultSuccess(
            message=result_message,
            gui_delegate=gui_delegate,
        )

    def _create_result_message(self, result_img: np.ndarray) -> str:
        
        txt = f'Segmentation done, with the following statistics:\n'
        
        for output_id, layer_sizes in enumerate(self._get_indexes_of_model_output_channels_to_create()):
            
            txt += f'Channels for output {output_id}:\n'
            
            unique, counts = np.unique(result_img[output_id], return_counts=True)
            counts_map = {}
            for i in range(len(unique)):
                counts_map[unique[i]] = counts[i]
                
            # # we cannot simply take image dimensions, because we may have irregular processing area from polygon
            number_of_pixels_in_processing_area = np.sum([counts_map[k] for k in counts_map.keys()])
            total_area = number_of_pixels_in_processing_area * self.params.resolution_m_per_px**2
            
            for channel_id in range(layer_sizes):
                # See note in the class description why are we adding/subtracting 1 here
                pixels_count = counts_map.get(channel_id, 0)
                area = pixels_count * self.params.resolution_m_per_px**2
                
                if total_area > 0 and not np.isnan(total_area) and not np.isinf(total_area):
                    area_percentage = area / total_area * 100
                else:
                    area_percentage = 0.0
                    # TODO
                    
                txt += f'\t- {self.model.get_channel_name(output_id, channel_id)}: area = {area:.2f} m^2 ({area_percentage:.2f} %)\n'

        return txt

    def _create_vlayer_from_mask_for_base_extent(self, mask_img) -> Callable:
        """ create vector layer with polygons from the mask image
        :return: function to be called in GUI thread
        """
        vlayers = []

        for output_id, layer_sizes in enumerate(self._get_indexes_of_model_output_channels_to_create()):
            output_vlayers = []
            for channel_id in range(layer_sizes):
                # See note in the class description why are we adding/subtracting 1 here
                local_mask_img = np.uint8(mask_img[output_id] == channel_id)

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

                layer_name = self.model.get_channel_name(output_id, channel_id)
                vlayer = QgsVectorLayer("multipolygon", layer_name, "memory")
                vlayer.setCrs(self.rlayer.crs())
                prov = vlayer.dataProvider()

                color = vlayer.renderer().symbol().color()
                OUTPUT_VLAYER_COLOR_TRANSPARENCY = 80
                color.setAlpha(OUTPUT_VLAYER_COLOR_TRANSPARENCY)
                vlayer.renderer().symbol().setColor(color)
                # TODO - add also outline for the layer (thicker black border)

                prov.addFeatures(features)
                vlayer.updateExtents()

                output_vlayers.append(vlayer)
                
            vlayers.append(output_vlayers)

        # accessing GUI from non-GUI thread is not safe, so we need to delegate it to the GUI thread
        def add_to_gui():
            group = QgsProject.instance().layerTreeRoot().insertGroup(0, 'model_output')
            
            if len(vlayers) == 1:
                for vlayer in vlayers[0]:
                    QgsProject.instance().addMapLayer(vlayer, False)
                    group.addLayer(vlayer)
            else:
                for i, output_vlayers in enumerate(vlayers):
                    output_group = group.insertGroup(0, f'output_{i}')
                    for vlayer in output_vlayers:
                        QgsProject.instance().addMapLayer(vlayer, False)
                        output_group.addLayer(vlayer)

        return add_to_gui

    def _process_tile(self, tile_img_batched: np.ndarray) -> np.ndarray:
        many_result = self.model.process(tile_img_batched)
        many_outputs = []

        for result in many_result:
            result[result < self.segmentation_parameters.pixel_classification__probability_threshold] = 0.0

            if len(result.shape) == 3:
                result = np.expand_dims(result, axis=1)

            if (result.shape[1] == 1):
                result = (result != 0).astype(int)
            else:
                shape = result.shape
                result = np.argmax(result, axis=1).reshape(shape[0], 1, shape[2], shape[3])

            assert len(result.shape) == 4
            assert result.shape[1] == 1

            many_outputs.append(result[:, 0])

        many_outputs = np.array(many_outputs).transpose((1, 0, 2, 3))
        
        return many_outputs
