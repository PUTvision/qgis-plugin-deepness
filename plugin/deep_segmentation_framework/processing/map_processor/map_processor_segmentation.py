import cv2
import numpy as np

from qgis.core import QgsVectorLayer
from qgis.core import QgsProject

from deep_segmentation_framework.common.processing_parameters.map_processing_parameters import ModelOutputFormat
from deep_segmentation_framework.processing import processing_utils
from deep_segmentation_framework.common.defines import IS_DEBUG
from deep_segmentation_framework.common.processing_parameters.segmentation_parameters import SegmentationParameters
from deep_segmentation_framework.processing.map_processor.map_processing_result import MapProcessingResult, \
    MapProcessingResultCanceled, MapProcessingResultSuccess
from deep_segmentation_framework.processing.map_processor.map_processor import MapProcessor
from deep_segmentation_framework.processing.map_processor.map_processor_with_model import MapProcessorWithModel

if IS_DEBUG:
    pass


class MapProcessorSegmentation(MapProcessorWithModel):
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

        result_message = self._create_result_message()
        return MapProcessingResultSuccess(result_message)

    def _create_result_message(self) -> str:
        return 'TODO Add here information about result!'

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

            vlayer = QgsVectorLayer("multipolygon", f"channel_{channel_id}", "memory")
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

