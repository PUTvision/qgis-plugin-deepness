""" This file implements map processing for detection model """

import re
import stat
from itertools import count
from turtle import distance
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
        for tile_img_batched, tile_params_batched in self.tiles_generator_batched():
            if self.isCanceled():
                return MapProcessingResultCanceled()

            bounding_boxes_in_tile_batched = self._process_tile(tile_img_batched, tile_params_batched)
            all_bounding_boxes += [d for det in bounding_boxes_in_tile_batched for d in det]

        if len(all_bounding_boxes) > 0:
            all_bounding_boxes_nms = self.remove_overlaping_detections(all_bounding_boxes, iou_threshold=self.detection_parameters.iou_threshold)
            all_bounding_boxes_restricted = self.limit_bounding_boxes_to_processed_area(all_bounding_boxes_nms)
        else:
            all_bounding_boxes_restricted = []

        gui_delegate = self._create_vlayer_for_output_bounding_boxes(all_bounding_boxes_restricted)

        result_message = self._create_result_message(all_bounding_boxes_restricted)
        self._all_detections = all_bounding_boxes_restricted
        return MapProcessingResultSuccess(
            message=result_message,
            gui_delegate=gui_delegate,
        )

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
        vlayers = []

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
                    contours, _ = cv2.findContours(det.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)

                    x_offset, y_offset = det.mask_offsets

                    if len(contours) > 0:
                        countur = contours[0]

                        corners = []
                        for point in countur:
                            corners.append(int(point[0][0]) + x_offset)
                            corners.append(int(point[0][1]) + y_offset)

                        mask_corners_pixels = cv2.convexHull(np.array(corners).reshape((-1, 2))).squeeze()

                        mask_corners_crs = processing_utils.transform_points_list_xy_to_target_crs(
                            points=mask_corners_pixels,
                            extent=self.extended_extent,
                            rlayer_units_per_pixel=self.rlayer_units_per_pixel,
                        )

                        feature = QgsFeature()
                        polygon_xy_vec_vec = [
                            mask_corners_crs
                        ]
                        geometry = QgsGeometry.fromPolygonXY(polygon_xy_vec_vec)
                        feature.setGeometry(geometry)
                        features.append(feature)

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

            vlayers.append(vlayer)

        # accessing GUI from non-GUI thread is not safe, so we need to delegate it to the GUI thread
        def add_to_gui():
            group = QgsProject.instance().layerTreeRoot().insertGroup(0, 'model_output')
            for vlayer in vlayers:
                QgsProject.instance().addMapLayer(vlayer, False)
                group.addLayer(vlayer)

        return add_to_gui

    @staticmethod
    def remove_overlaping_detections(bounding_boxes: List[Detection], iou_threshold: float) -> List[Detection]:
        bboxes = []
        probs = []
        for det in bounding_boxes:
            bboxes.append(det.get_bbox_xyxy())
            probs.append(det.conf)

        bboxes = np.array(bboxes)
        probs = np.array(probs)

        import time

        start = time.time()
        pick_ids = Detector.non_max_suppression_fast(bboxes, probs, iou_threshold)
        stop = time.time()
        print(f"Time: {stop - start}")

        filtered_bounding_boxes = [x for i, x in enumerate(bounding_boxes) if i in pick_ids]
        filtered_bounding_boxes = sorted(filtered_bounding_boxes, reverse=True)

        start = time.time()
        pick_ids_kde = MapProcessorDetection.non_max_kdtree(filtered_bounding_boxes, iou_threshold)
        stop = time.time()
        print(f"Time: {stop - start}")

        filtered_bounding_boxes = [x for i, x in enumerate(filtered_bounding_boxes) if i in pick_ids_kde]

        return filtered_bounding_boxes

    @staticmethod
    def non_max_kdtree(bounding_boxes: List[Detection], iou_threshold: float) -> List[int]:
        """ Remove overlapping bounding boxes using kdtree

        :param bounding_boxes: List of bounding boxes in (xyxy format)
        :param iou_threshold: Threshold for intersection over union
        :return: Pick ids to keep
        """
        from scipy.spatial import cKDTree

        centers = np.array([det.get_bbox_center() for det in bounding_boxes])

        kdtree = cKDTree(centers)
        pick_ids = set()
        removed_ids = set()

        for i, bbox in enumerate(bounding_boxes):
            if i in removed_ids:
                continue

            _, indices = kdtree.query(bbox.get_bbox_center(), k=10)

            for j in indices:
                if j in removed_ids:
                    continue

                if i == j:
                    continue

                iou = bbox.bbox.calculate_intersection_over_smaler_area(bounding_boxes[j].bbox)
                
                if iou > iou_threshold:
                    removed_ids.add(j)

            pick_ids.add(i)
            
        return pick_ids

    @staticmethod
    def convert_bounding_boxes_to_absolute_positions(bounding_boxes_relative: List[Detection],
                                                     tile_params: TileParams):
        for det in bounding_boxes_relative:
            det.convert_to_global(offset_x=tile_params.start_pixel_x, offset_y=tile_params.start_pixel_y)

    def _process_tile(self, tile_img: np.ndarray, tile_params_batched: List[TileParams]) -> np.ndarray:
        bounding_boxes_batched: List[Detection] = self.model.process(tile_img)

        for bounding_boxes, tile_params in zip(bounding_boxes_batched, tile_params_batched):
            self.convert_bounding_boxes_to_absolute_positions(bounding_boxes, tile_params)

        return bounding_boxes_batched
