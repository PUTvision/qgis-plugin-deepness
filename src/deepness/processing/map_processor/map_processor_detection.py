""" This file implements map processing for detection model """

from typing import List

import cv2
import numpy as np
from qgis.core import QgsFeature, QgsGeometry, QgsProject, QgsVectorLayer

from deepness.common.processing_parameters.detection_parameters import DetectionParameters, DetectorType
from deepness.processing import processing_utils
from deepness.processing.map_processor.map_processing_result import (MapProcessingResult, MapProcessingResultCanceled,
                                                                     MapProcessingResultSuccess)
from deepness.processing.map_processor.map_processor_with_model import MapProcessorWithModel
from deepness.processing.models.detector import Detection, Detector
from deepness.processing.tile_params import TileParams


class MapProcessorDetection(MapProcessorWithModel):
    """
    MapProcessor specialized for detecting objects (where there is a finite list of detected objects
    of different classes, which area (bounding boxes) may overlap)
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
        self.model.set_model_type_param(model_type=params.detector_type)
        self._all_detections = None

    def get_all_detections(self) -> List[Detection]:
        return self._all_detections

    def _run(self) -> MapProcessingResult:
        all_bounding_boxes = []  # type: List[Detection]
        for tile_img, tile_params in self.tiles_generator():
            if self.isCanceled():
                return MapProcessingResultCanceled()

            bounding_boxes_in_tile = self._process_tile(tile_img, tile_params)
            all_bounding_boxes += bounding_boxes_in_tile

        if len(all_bounding_boxes) > 0:
            all_bounding_boxes_suppressed = self.apply_non_maximum_suppression(all_bounding_boxes)
            all_bounding_boxes_restricted = self.limit_bounding_boxes_to_processed_area(all_bounding_boxes_suppressed)
        else:
            all_bounding_boxes_restricted = []

        self._create_vlayer_for_output_bounding_boxes(all_bounding_boxes_restricted)

        result_message = self._create_result_message(all_bounding_boxes_restricted)
        self._all_detections = all_bounding_boxes_restricted
        return MapProcessingResultSuccess(result_message)

    def limit_bounding_boxes_to_processed_area(self, bounding_boxes: List[Detection]) -> List[Detection]:
        """
        Limit all bounding boxes to the constrained area that we process.
        E.g. if we are detecting peoples in a circle, we don't want to count peoples in the entire rectangle

        :return:
        """
        bounding_boxes_restricted = []
        for det in bounding_boxes:
            # if bounding box is not in the area_mask_img (at least in some percentage) - remove it

            if self.area_mask_img is not None:
                det_slice = det.bbox.get_slice()
                area_subimg = self.area_mask_img[det_slice]
                pixels_in_area = np.count_nonzero(area_subimg)
            else:
                det_bounding_box = det.bbox
                pixels_in_area = self.base_extent_bbox_in_full_image.calculate_overlap_in_pixels(det_bounding_box)
            total_pixels = det.bbox.get_area()
            coverage = pixels_in_area / total_pixels
            if coverage > 0.5:  # some arbitrary value, 50% seems reasonable
                bounding_boxes_restricted.append(det)

        return bounding_boxes_restricted

    def _create_result_message(self, bounding_boxes: List[Detection]) -> str:
        channels = self._get_indexes_of_model_output_channels_to_create()

        counts_mapping = {}
        total_counts = 0
        for channel_id in channels:
            filtered_bounding_boxes = [det for det in bounding_boxes if det.clss == channel_id]
            counts = len(filtered_bounding_boxes)
            counts_mapping[channel_id] = counts
            total_counts += counts

        txt = f'Detection done for {len(channels)} model output classes, with the following statistics:\n'
        for channel_id in channels:
            counts = counts_mapping[channel_id]

            if total_counts:
                counts_percentage = counts / total_counts * 100
            else:
                counts_percentage = 0

            txt += f' - {self.model.get_channel_name(channel_id)}: counts = {counts} ({counts_percentage:.2f} %)\n'

        return txt

    def _create_vlayer_for_output_bounding_boxes(self, bounding_boxes: List[Detection]):
        group = QgsProject.instance().layerTreeRoot().insertGroup(0, 'model_output')

        for channel_id in self._get_indexes_of_model_output_channels_to_create():
            filtered_bounding_boxes = [det for det in bounding_boxes if det.clss == channel_id]
            print(f'Detections for class {channel_id}: {len(filtered_bounding_boxes)}')

            features = []
            for det in filtered_bounding_boxes:
                if det.mask is None:
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
                else:
                    contours, hierarchy = cv2.findContours(det.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours = processing_utils.transform_contours_yx_pixels_to_target_crs(
                        contours=contours,
                        extent=self.base_extent,
                        rlayer_units_per_pixel=self.rlayer_units_per_pixel)

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

    def apply_non_maximum_suppression(self, bounding_boxes: List[Detection]) -> List[Detection]:
        bboxes = []
        probs = []
        for det in bounding_boxes:
            bboxes.append(det.get_bbox_xyxy())
            probs.append(det.conf)

        bboxes = np.array(bboxes)
        probs = np.array(probs)

        pick_ids = self.model.non_max_suppression_fast(bboxes, probs, self.detection_parameters.iou_threshold)

        filtered_bounding_boxes = [x for i, x in enumerate(bounding_boxes) if i in pick_ids]

        if self.detection_parameters.remove_overlapping_detections:
            filtered_bounding_boxes = sorted(filtered_bounding_boxes, reverse=True)

            to_remove = []
            for i in range(len(filtered_bounding_boxes)):
                if i in to_remove:
                    continue
                for j in range(i + 1, len(filtered_bounding_boxes)):
                    if j in to_remove:
                        continue
                    if i != j:
                        if filtered_bounding_boxes[i].bbox.calculate_intersection_over_smaler_area(
                                filtered_bounding_boxes[j].bbox) > self.detection_parameters.iou_threshold:
                            to_remove.append(j)

            filtered_bounding_boxes = [x for i, x in enumerate(filtered_bounding_boxes) if i not in to_remove]

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
