import copy
from typing import List

import cv2
import numpy as np
from qgis.core import QgsVectorLayer, QgsProject, QgsGeometry, QgsFeature

from deep_segmentation_framework.common.processing_parameters.detection_parameters import DetectionParameters
from deep_segmentation_framework.common.defines import IS_DEBUG
from deep_segmentation_framework.processing import processing_utils
from deep_segmentation_framework.processing.map_processor.map_processor import MapProcessor
from deep_segmentation_framework.processing.map_processor.map_processor_with_model import MapProcessorWithModel
from deep_segmentation_framework.processing.models.detector import Detector
from deep_segmentation_framework.processing.tile_params import TileParams
from deep_segmentation_framework.processing.models.detector import Detection

if IS_DEBUG:
    pass


class MapProcessorDetection(MapProcessorWithModel):
    """
    Process the entire map for the detection models, which produce bounding boxes
    """

    def __init__(self,
                 params: DetectionParameters,
                 **kwargs):
        super().__init__(
            params=params,
            model=params.model,
            **kwargs)
        self.detection_parameters = params
        self.model = params.model  # type: Detector
        self.model.set_inference_params(
            confidence=params.confidence,
            iou_threshold=params.iou_threshold
        )

    def _run(self):
        all_bounding_boxes = []  # type: List[Detection]
        for tile_img, tile_params in self.tiles_generator():
            if self.isCanceled():
                return False

            bounding_boxes_in_tile = self._process_tile(tile_img, tile_params)
            all_bounding_boxes += bounding_boxes_in_tile

        if len(all_bounding_boxes) > 0:
            all_bounding_boxes_suppressed = self.apply_non_maximum_suppression(all_bounding_boxes)

            all_bounding_boxes_restricted = self.limit_bounding_boxes_to_processed_area(all_bounding_boxes_suppressed)
        else:
            all_bounding_boxes_restricted = []

        self._create_vlayer_for_output_bounding_boxes(all_bounding_boxes_restricted)

        return True

    def limit_bounding_boxes_to_processed_area(self, bounding_boxes):
        """
        Limit all bounding boxes to the constrained area that we process.
        E.g. if we are detecting peoples in a circle, we don't want to count peoples in the entire rectangle

        # TODO! implement!

        :return:
        """

        # self.area_mask_img = processing_utils.create_area_mask_image(
        #     vlayer_mask=self.vlayer_mask,
        #     extended_extent=self.extended_extent,
        #     rlayer_units_per_pixel=self.rlayer_units_per_pixel,
        #     image_shape_yx=[self.img_size_y_pixels, self.img_size_x_pixels])

        # if bounding box is not in the area_mask_img (at least in some percentage) - remove it
        return bounding_boxes

    def _create_vlayer_for_output_bounding_boxes(self, bounding_boxes: List[Detection]):
        group = QgsProject.instance().layerTreeRoot().insertGroup(0, 'model_output')

        for channel_id in self._get_indexes_of_model_output_channels_to_create():
            filtered_bounding_boxes = [det for det in bounding_boxes if det.clss == channel_id]
            print(f'Detections for class {channel_id}: {len(filtered_bounding_boxes)}')

            features = []
            for det in filtered_bounding_boxes:
                bbox_corners_pixels = det.bbox.get_4_corners()
                bbox_corners_crs = processing_utils.transform_points_list_xy_to_target_crs(
                    points=bbox_corners_pixels,
                    extent=self.extended_extent,
                    rlayer_units_per_pixel=self.rlayer_units_per_pixel,
                )
                feature = QgsFeature()
                polygon_xy_vec_vec = [
                    bbox_corners_crs
                ]
                geometry = QgsGeometry.fromPolygonXY(polygon_xy_vec_vec)
                feature.setGeometry(geometry)
                features.append(feature)

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

    def apply_non_maximum_suppression(self, bounding_boxes: List[Detection]) -> List[Detection]:
        bboxes = []
        for det in bounding_boxes:
            bboxes.append(det.get_bbox_xyxy())

        bboxes = np.stack(bboxes, axis=0)
        pick_ids = self.model.non_max_suppression_fast(bboxes, self.detection_parameters.confidence)

        filtered_bounding_boxes = [x for i, x in enumerate(bounding_boxes) if i in pick_ids]

        return filtered_bounding_boxes

    @staticmethod
    def convert_bounding_boxes_to_absolute_positions(bounding_boxes_relative: List[Detection],
                                                     tile_params: TileParams):
        for det in bounding_boxes_relative:
            det.convert_to_global(offset_x=tile_params.start_pixel_x, offset_y=tile_params.start_pixel_y)

    def _process_tile(self, tile_img: np.ndarray, tile_params: TileParams) -> np.ndarray:
        bounding_boxes: List[Detection] = self.model.process(tile_img)
        self.convert_bounding_boxes_to_absolute_positions(bounding_boxes, tile_params)
        return bounding_boxes

