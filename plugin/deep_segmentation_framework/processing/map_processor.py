import copy
import logging
import time
from typing import Optional, List, Tuple

import numpy as np
import cv2
import onnxruntime as ort

from qgis.PyQt.QtCore import pyqtSignal
from qgis.core import QgsWkbTypes
from qgis.core import QgsFeature, QgsGeometry, QgsVectorLayer, QgsPointXY
from qgis.gui import QgsMapCanvas
from qgis.core import QgsRasterLayer
from qgis.core import QgsUnitTypes
from qgis.core import QgsRectangle
from qgis.core import QgsMessageLog
from qgis.core import QgsApplication
from qgis.core import QgsTask
from qgis.core import QgsProject
from qgis.core import QgsCoordinateTransform
from qgis.gui import QgisInterface
from qgis.core import Qgis
import qgis

from deep_segmentation_framework.processing import processing_utils, extent_utils
from deep_segmentation_framework.common.defines import PLUGIN_NAME, LOG_TAB_NAME, IS_DEBUG
from deep_segmentation_framework.common.inference_parameters import InferenceParameters, ProcessedAreaType
from deep_segmentation_framework.processing.model_wrapper import ModelWrapper
from deep_segmentation_framework.processing.processing_utils import BoundingBox
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
                 inference_parameters: InferenceParameters):
        """

        :param rlayer: Raster layer whihc is being processed
        :param map_canvas: active map canvas (in the GUI), required if processing visible map area
        :param inference_parameters: see InferenceParameters
        """
        QgsTask.__init__(self, self.__class__.__name__)
        self.rlayer = rlayer
        self.vlayer_mask = vlayer_mask
        if vlayer_mask:
            assert vlayer_mask.crs() == self.rlayer.crs()  # should be set in higher layer
        self.inference_parameters = inference_parameters

        self.stride_px = self.inference_parameters.processing_stride_px  # stride in pixels
        self.rlayer_units_per_pixel = processing_utils.convert_meters_to_rlayer_units(
            self.rlayer, self.inference_parameters.resolution_m_per_px)  # number of rlayer units for one tile pixel

        # extent in which the actual required area is contained, without additional extensions, rounded to rlayer grid
        self.base_extent = extent_utils.calculate_base_processing_extent_in_rlayer_crs(
            map_canvas=map_canvas,
            rlayer=self.rlayer,
            vlayer_mask=self.vlayer_mask,
            inference_parameters=self.inference_parameters)

        # extent which should be used during model inference, as it includes extra margins to have full tiles,
        # rounded to rlayer grid
        self.extended_extent = extent_utils.calculate_extended_processing_extent(
            base_extent=self.base_extent,
            rlayer=self.rlayer,
            inference_parameters=self.inference_parameters,
            rlayer_units_per_pixel=self.rlayer_units_per_pixel)

        # processed rlayer dimensions (for extended_extent)
        self.img_size_x_pixels = round(self.extended_extent.width() / self.rlayer_units_per_pixel)  # how many columns (x)
        self.img_size_y_pixels = round(self.extended_extent.height() / self.rlayer_units_per_pixel)  # how many rows (y)

        # Coordinate of base image withing extended image (images for base_extent and extended_extent)
        self.base_extent_bbox_in_full_image = self._calculate_base_extent_bbox_in_full_image(self.img_size_y_pixels)

        # Number of tiles in x and y dimensions which will be used during processing
        # As we are using "extended_extent" this should divide without any rest
        self.x_bins_number = round((self.img_size_x_pixels - self.inference_parameters.tile_size_px)
                                   / self.stride_px) + 1
        self.y_bins_number = round((self.img_size_y_pixels - self.inference_parameters.tile_size_px)
                                   / self.stride_px) + 1

        self.model_wrapper = ModelWrapper(model_file_path=inference_parameters.model_file_path)

    def run(self):
        print('run...')
        return self._process()

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

    def transform_polygon_with_rings_epsg_to_extended_xy_pixels(self, polygons: List[List[QgsPointXY]]) -> List[List[Tuple]]:
        """
        Transform coordinates polygons to pixels contours (with cv2 format), in base_extent pixels system
        :param polygons: List of tuples with two lists each (x and y points respoectively)
        :return: 2D contours list
        """
        xy_pixel_contours = []
        for polygon in polygons:
            xy_pixel_contour = []

            x_min_epsg = self.extended_extent.xMinimum()
            y_min_epsg = self.extended_extent.yMinimum()
            y_max_pixel = self.img_size_y_pixels - 1  # -1 to have the max pixel, not shape
            for point_epsg in polygon:
                x_epsg, y_epsg = point_epsg
                x = round((x_epsg - x_min_epsg) / self.rlayer_units_per_pixel)
                y = y_max_pixel - round((y_epsg - y_min_epsg) / self.rlayer_units_per_pixel)
                # NOTE: here we can get pixels +-1 values, because we operate on already rounded bounding boxes
                xy_pixel_contour.append((x, y))

            # Values:
            # self.base_extent.height() / self.rlayer_units_per_pixel, self.base_extent.width() / self.rlayer_units_per_pixel
            # are not integers, because extents are aligned to grid, not pixels resolution

            xy_pixel_contours.append(np.asarray(xy_pixel_contour))
        return xy_pixel_contours

    def _create_area_mask_image(self, vlayer_mask) -> Optional[np.ndarray]:
        """
        Mask determining area to process (within extended_extent coordinates)
        None if no mask layer provided.
        """

        if vlayer_mask is None:
            return None
        img = np.zeros(shape=[self.img_size_y_pixels, self.img_size_x_pixels], dtype=np.uint8)
        features = vlayer_mask.getFeatures()

        # see https://docs.qgis.org/3.22/en/docs/pyqgis_developer_cookbook/vector.html#iterating-over-vector-layer
        for feature in features:
            print("Feature ID: ", feature.id())
            geom = feature.geometry()
            geom_single_type = QgsWkbTypes.isSingleType(geom.wkbType())

            if geom.type() == QgsWkbTypes.PointGeometry:
                logging.warning("Point geometry not supported!")
            elif geom.type() == QgsWkbTypes.LineGeometry:
                logging.warning("Line geometry not supported!")
            elif geom.type() == QgsWkbTypes.PolygonGeometry:
                polygons = []
                if geom_single_type:
                    polygon = geom.asPolygon()  # polygon with rings
                    polygons.append(polygon)
                else:
                    polygons = geom.asMultiPolygon()

                for polygon_with_rings in polygons:
                    polygon_with_rings_xy = self.transform_polygon_with_rings_epsg_to_extended_xy_pixels(polygon_with_rings)
                    # first polygon is actual polygon
                    cv2.fillPoly(img, pts=polygon_with_rings_xy[:1], color=255)
                    if len(polygon_with_rings_xy) > 1:  # further polygons are rings
                        cv2.fillPoly(img, pts=polygon_with_rings_xy[1:], color=0)
            else:
                print("Unknown or invalid geometry")

        return img

    def _process(self):
        total_tiles = self.x_bins_number * self.y_bins_number
        final_shape_px = (self.img_size_y_pixels, self.img_size_x_pixels)
        full_result_img = np.zeros(final_shape_px, np.uint8)
        mask_img = self._create_area_mask_image(self.vlayer_mask)

        for y_bin_number in range(self.y_bins_number):
            for x_bin_number in range(self.x_bins_number):
                if self.isCanceled():
                    return False

                tile_no = y_bin_number * self.x_bins_number + x_bin_number
                progress = tile_no / total_tiles * 100
                self.setProgress(progress)
                print(f" Processing tile {tile_no} / {total_tiles} [{progress:.2f}%]")

                tile_params = TileParams(x_bin_number=x_bin_number, y_bin_number=y_bin_number,
                                         x_bins_number=self.x_bins_number, y_bins_number=self.y_bins_number,
                                         inference_parameters=self.inference_parameters,
                                         processing_extent=self.extended_extent,
                                         rlayer_units_per_pixel=self.rlayer_units_per_pixel)

                if not tile_params.is_tile_within_mask(mask_img):
                    continue  # tile outside of mask - to be skipped

                tile_img = processing_utils.get_tile_image(self.rlayer, tile_params.extent, self.inference_parameters)

                # TODO add support for smaller

                tile_result = self._process_tile(tile_img)
                # plt.figure(); plt.imshow(tile_img); plt.show(block=False); plt.pause(0.001)
                # self._show_image(tile_result)
                self._set_mask_on_full_img(tile_result=tile_result,
                                           full_result_img=full_result_img,
                                           tile_params=tile_params)

        full_result_img = processing_utils.erode_dilate_image(img=full_result_img,
                                                              inference_parameters=self.inference_parameters)
        # plt.figure(); plt.imshow(full_result_img); plt.show(block=False); plt.pause(0.001)
        result_img = self.limit_extended_extent_image_to_base_extent_with_mask(full_img=full_result_img,
                                                                               mask_img=mask_img)
        self._create_vlayer_from_mask_for_base_extent(result_img)
        return True

    def _calculate_base_extent_bbox_in_full_image(self, image_size_y) -> BoundingBox:
        """

        :param image_size_y: Number of y pixels (rows)
        :return:
        """
        base_extent = self.base_extent
        extended_extent = self.extended_extent
        rlayer_units_per_pixel = self.rlayer_units_per_pixel

        # should round without a rest anyway, as extends are aligned to rlayer grid
        base_extent_bbox_in_full_image = BoundingBox(
            x_min=round((base_extent.xMinimum() - extended_extent.xMinimum()) / rlayer_units_per_pixel),
            y_min=image_size_y - 1 - round((base_extent.yMaximum() - extended_extent.yMinimum()) / rlayer_units_per_pixel - 1),
            x_max=round((base_extent.xMaximum() - extended_extent.xMinimum()) / rlayer_units_per_pixel) - 1,
            y_max=image_size_y - 1 - round((base_extent.yMinimum() - extended_extent.yMinimum()) / rlayer_units_per_pixel),
        )
        return base_extent_bbox_in_full_image

    def limit_extended_extent_image_to_base_extent_with_mask(self, full_img, mask_img: Optional[np.ndarray]):
        """
        Limit an image which is for extended_extent to the base_extent image.
        If a limiting polygon was used for processing, it will be also applied.
        :param full_img:
        :param mask_img: Image with processed area mask (if a constrained area used)
        :return:
        """

        # TODO look for some inplace operation to save memory
        # cv2.copyTo(src=full_img, mask=mask_img, dst=full_img)  # this doesn't work due to implementation details
        full_img = cv2.copyTo(src=full_img, mask=mask_img)

        b = self.base_extent_bbox_in_full_image
        result_img = full_img[b.y_min:b.y_max+1, b.x_min:b.x_max+1]
        return result_img

    def _set_mask_on_full_img(self, full_result_img, tile_result, tile_params: TileParams):
        roi_slice_on_full_image = tile_params.get_slice_on_full_image_for_copying()
        roi_slice_on_tile_image = tile_params.get_slice_on_tile_image_for_copying(roi_slice_on_full_image)
        full_result_img[roi_slice_on_full_image] = tile_result[roi_slice_on_tile_image]

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
        # TODO - create proper mapping for channels (layer channels to model channels)
        # Handle RGB, RGBA properly
        # TODO check if we have RGB or RGBA

        # thresholding on one channel
        # tile_img = copy.copy(tile_img)
        # tile_img = tile_img[:, :, 1]
        # tile_img[tile_img <= 100] = 0
        # tile_img[tile_img > 100] = 255
        # result = tile_img

        # thresholding on Red channel (max value - with manually drawn dots on fotomap)
        tile_img = copy.copy(tile_img)
        tile_img = tile_img[:, :, 0]
        tile_img[tile_img < 255] = 0
        tile_img[tile_img >= 255] = 255
        result = tile_img
        return result

        result = self.model_wrapper.process(tile_img)
        return result

