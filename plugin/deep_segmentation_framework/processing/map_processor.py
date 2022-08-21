import copy
import time

import numpy as np
import cv2
import onnxruntime as ort

from qgis.PyQt.QtCore import pyqtSignal
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
from deep_segmentation_framework.processing.tile_params import TileParams

if IS_DEBUG:
    from matplotlib import pyplot as plt


class MapProcessor(QgsTask):
    finished_signal = pyqtSignal(str)  # error message if finished with error, empty string otherwise
    show_img_signal = pyqtSignal(object, str)  # request to show an image. Params: (image, window_name)

    def __init__(self,
                 rlayer: QgsRasterLayer,
                 map_canvas: QgsMapCanvas,
                 inference_parameters: InferenceParameters):
        """

        :param rlayer: Raster layer whihc is being processed
        :param map_canvas: active map canvas (in the GUI), required if processing visible map area
        :param inference_parameters: see InferenceParameters
        """
        QgsTask.__init__(self, self.__class__.__name__)
        self.rlayer = rlayer
        self.inference_parameters = inference_parameters

        self.stride_px = self.inference_parameters.processing_stride_px  # stride in pixels
        self.rlayer_units_per_pixel = processing_utils.convert_meters_to_rlayer_units(
            self.rlayer, self.inference_parameters.resolution_m_per_px)  # number of rlayer units for one tile pixel

        # extent in which the actual required area is contained, without additional extensions
        self.base_extent = self._calculate_base_processing_extent_in_rlayer_crs(
            map_canvas=map_canvas
        )  # type: QgsRectangle

        # extent which should be used during model inference, as it includes extra margins to have full tiles
        self.extended_extent = self._calculate_extended_processing_extent(
            base_extent=self.base_extent)

        # processed rlayer dimensions (for extended_extent)
        self.img_size_x_pixels = round(self.extended_extent.width() / self.rlayer_units_per_pixel)
        self.img_size_y_pixels = round(self.extended_extent.height() / self.rlayer_units_per_pixel)

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

    def is_busy(self):
        return True

    def _show_image(self, img, window_name='img'):
        self.show_img_signal.emit(img, window_name)

    def _process(self):
        total_tiles = self.x_bins_number * self.y_bins_number
        final_shape_px = (self.img_size_y_pixels, self.img_size_x_pixels)
        full_result_img = np.zeros(final_shape_px, np.uint8)

        if total_tiles < 1:
            raise Exception("TODO! Add support for partial tiles!")
        # TODO - add support for to small images - padding for the last bin
        # (and also bins_number calculation, to have at least one)

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
        self._create_vlayer_from_mask(full_result_img)
        return True

    def _set_mask_on_full_img(self, full_result_img, tile_result, tile_params: TileParams):
        roi_slice_on_full_image = tile_params.get_slice_on_full_image_for_copying()
        roi_slice_on_tile_image = tile_params.get_slice_on_tile_image_for_copying(roi_slice_on_full_image)
        full_result_img[roi_slice_on_full_image] = tile_result[roi_slice_on_tile_image]

    def _create_vlayer_from_mask(self, mask_img):
        # create vector layer with polygons from the mask image
        contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = self.transform_contours_yx_pixels_to_target_crs(contours)
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

    def transform_contours_yx_pixels_to_target_crs(self, polygons):
        x_left = self.extended_extent.xMinimum()
        y_upper = self.extended_extent.yMaximum()

        polygons_crs = []
        for polygon_3d in polygons:
            # https://stackoverflow.com/questions/33458362/opencv-findcontours-why-do-we-need-a-vectorvectorpoint-to-store-the-cont
            polygon = polygon_3d.squeeze(axis=1)

            polygon_crs = []
            for i in range(len(polygon)):
                yx_px = polygon[i]
                x_crs = yx_px[0] * self.rlayer_units_per_pixel + x_left
                y_crs = -(yx_px[1] * self.rlayer_units_per_pixel - y_upper)
                polygon_crs.append(QgsPointXY(x_crs, y_crs))
            polygons_crs.append(polygon_crs)
        return polygons_crs

    def _calculate_base_processing_extent_in_rlayer_crs(self, map_canvas: QgsMapCanvas):
        """
        Determine the Base Extent of processing (Extent (rectangle) in which the actual required area is contained)
        :param map_canvas: active map canvas (in the GUI), required if processing visible map area
        """
        rlayer_extent = self.rlayer.extent()
        processed_area_type = self.inference_parameters.processed_area_type

        if processed_area_type == ProcessedAreaType.ENTIRE_LAYER:
            expected_extent = rlayer_extent
        elif processed_area_type == ProcessedAreaType.FROM_POLYGONS:
            mask_layer_name = self.inference_parameters.mask_layer_name
            assert mask_layer_name is not None
            active_extent_in_mask_layer_crs = QgsProject.instance().mapLayersByName(mask_layer_name)[0]
            active_extent = active_extent_in_mask_layer_crs.getGeometry(0)
            active_extent.convertToSingleType()
            active_extent = active_extent.boundingBox()

            t = QgsCoordinateTransform()
            t.setSourceCrs(active_extent_in_mask_layer_crs.sourceCrs())
            t.setDestinationCrs(self.rlayer.crs())
            expected_extent = t.transform(active_extent)
        elif processed_area_type == ProcessedAreaType.VISIBLE_PART:
            # transform visible extent from mapCanvas CRS to layer CRS
            active_extent_in_canvas_crs = map_canvas.extent()
            canvas_crs = map_canvas.mapSettings().destinationCrs()
            t = QgsCoordinateTransform()
            t.setSourceCrs(canvas_crs)
            t.setDestinationCrs(self.rlayer.crs())
            expected_extent = t.transform(active_extent_in_canvas_crs)
        else:
            raise Exception("Invalid processed are type!")

        expected_extent = extent_utils.round_extent_to_rlayer_grid(extent=expected_extent, rlayer=self.rlayer)
        base_extent = expected_extent.intersect(rlayer_extent)

        return base_extent

    def _calculate_extended_processing_extent(self, base_extent: QgsRectangle):
        # first try to add pixels at every border - same as half-overlap for other tiles
        additional_pixels = self.inference_parameters.processing_overlap_px // 2
        additional_pixels_in_units = additional_pixels * self.rlayer_units_per_pixel

        tmp_extent = QgsRectangle(
            base_extent.xMinimum() - additional_pixels_in_units,
            base_extent.yMinimum() - additional_pixels_in_units,
            base_extent.xMaximum() + additional_pixels_in_units,
            base_extent.yMaximum() + additional_pixels_in_units,
        )
        tmp_extent = tmp_extent.intersect(self.rlayer.extent())

        # then add borders to have the extent be equal to  N * stride + tile_size, where N is a natural number
        tile_size_px = self.inference_parameters.tile_size_px
        stride_px = self.stride_px  # stride in pixels

        current_x_pixels = round(tmp_extent.width() / self.rlayer_units_per_pixel)
        if current_x_pixels <= tile_size_px:
            missing_pixels_x = tile_size_px - current_x_pixels  # just one tile
        else:
            pixels_in_last_stride_x = (current_x_pixels - tile_size_px) % stride_px
            missing_pixels_x = (stride_px - pixels_in_last_stride_x) % stride_px

        current_y_pixels = round(tmp_extent.height() / self.rlayer_units_per_pixel)
        if current_y_pixels <= tile_size_px:
            missing_pixels_y = tile_size_px - current_y_pixels  # just one tile
        else:
            pixels_in_last_stride_y = (current_y_pixels - tile_size_px) % stride_px
            missing_pixels_y = (stride_px - pixels_in_last_stride_y) % stride_px

        missing_pixels_x_in_units = missing_pixels_x * self.rlayer_units_per_pixel
        missing_pixels_y_in_units = missing_pixels_y * self.rlayer_units_per_pixel
        tmp_extent.setXMaximum(tmp_extent.xMaximum() + missing_pixels_x_in_units)
        tmp_extent.setYMaximum(tmp_extent.yMaximum() + missing_pixels_y_in_units)

        extended_extent = tmp_extent
        return extended_extent

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

