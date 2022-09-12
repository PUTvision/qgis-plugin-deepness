from typing import List

import cv2
import numpy as np
from qgis._core import QgsVectorLayer, QgsProject

from deep_segmentation_framework.common.processing_parameters.detection_parameters import DetectionParameters
from deep_segmentation_framework.common.defines import IS_DEBUG
from deep_segmentation_framework.processing import processing_utils
from deep_segmentation_framework.processing.map_processor.map_processor import MapProcessor
from deep_segmentation_framework.processing.tile_params import TileParams
from deep_segmentation_framework.processing.models.detector import Detection

if IS_DEBUG:
    pass


class MapProcessorDetection(MapProcessor):
    """
    Process the entire map for the detection models, which produce bounding boxes
    """

    def __init__(self,
                 params: DetectionParameters,
                 **kwargs):
        super().__init__(
            params=params,
            **kwargs)
        self.detection_parameters = params
        self.model = params.model

    def _run(self):
        all_bounding_boxes = []  # type: List[...]
        for tile_img, tile_params in self.tiles_generator():
            if self.isCanceled():
                return False

            bounding_boxes_in_tile = self._process_tile(tile_img, tile_params)
            all_bounding_boxes += bounding_boxes_in_tile

        all_bounding_boxes_suppressed = self.apply_non_maximum_supression(all_bounding_boxes)

        all_bounding_boxes_restricted = self.limit_bounding_boxes_to_processed_area(all_bounding_boxes_suppressed)

        self._create_vlayer_for_output_bounding_boxes(all_bounding_boxes_restricted)

        return True

    def limit_bounding_boxes_to_processed_area(self, bounding_boxes):
        """
        Limit all bounding boxes to the constrained area that we process.
        E.g. if we are detecting peoples in a circle, we don't want to count peoples in the entire rectangle
        :return:
        """
        self.area_mask_img = processing_utils.create_area_mask_image(
            vlayer_mask=self.vlayer_mask,
            extended_extent=self.extended_extent,
            rlayer_units_per_pixel=self.rlayer_units_per_pixel,
            image_shape_yx=[self.img_size_y_pixels, self.img_size_x_pixels])

        # if bounding box is not in the area_mask_img (at least in some percentage) - remove it
        return bounding_boxes

    def _create_vlayer_for_output_bounding_boxes(self, bounding_boxes):
        group = QgsProject.instance().layerTreeRoot().addGroup('model_output')

        number_of_output_classes = self.detection_parameters.model.get_number_of_output_channels()

        for channel_id in range(0, number_of_output_classes):
            local_mask_img = np.zeros((self.img_size_y_pixels, self.img_size_x_pixels), dtype=np.uint8)

            filtered_bounding_boxes = [det for det in bounding_boxes if det.clss == channel_id]
            for det in filtered_bounding_boxes:
                cv2.rectangle(local_mask_img, det.bbox.left_upper, det.bbox.right_down, 1, -1)

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

    def apply_non_maximum_supression(self, bounding_boxes: List[Detection]) -> List[Detection]:
        bboxes = []
        for det in bounding_boxes:
            bboxes.append(det.get_bbox_xyxy())

        bboxes = np.stack(bboxes, axis=0)
        pick_ids = self.model.non_max_suppression_fast(bboxes, self.model.iou_threshold)

        filtered_bounding_boxes = [x for i, x in enumerate(bounding_boxes) if i in pick_ids]

        return filtered_bounding_boxes

    @staticmethod
    def convert_bounding_boxes_to_absolute_positions(bounding_boxes_relative: List[Detection], tile_params: TileParams) -> List[Detection]:
        for det in bounding_boxes_relative:
            det.convert_to_global(offset_x=tile_params.start_pixel_x, offset_y=tile_params.start_pixel_y)

        return bounding_boxes_relative

    def _process_tile(self, tile_img: np.ndarray, tile_params: TileParams) -> np.ndarray:
        # TODO - create proper mapping for output channels
        bounding_boxes_relative: List[Detection] = self.model.process(tile_img)

        bounding_boxes_absolute_positions = self.convert_bounding_boxes_to_absolute_positions(
            bounding_boxes_relative, tile_params)

        return bounding_boxes_absolute_positions

