import copy
import time

import numpy as np
import cv2

from qgis.PyQt.QtCore import pyqtSignal
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

from ..common.defines import PLUGIN_NAME, LOG_TAB_NAME
from ..common.inference_parameters import InferenceParameters


class TileParams:
    def __init__(self,
                 x_bin_number,
                 y_bin_number,
                 x_bins_number,
                 y_bins_number,
                 inference_parameters: InferenceParameters,
                 px_in_rlayer_units,
                 file_extent):
        self.x_bin_number = x_bin_number
        self.y_bin_number = y_bin_number
        self.x_bins_number = x_bins_number
        self.y_bins_number = y_bins_number
        self.stride_px = inference_parameters.processing_stride_px
        self.start_pixel_x = x_bin_number * self.stride_px
        self.start_pixel_y = y_bin_number * self.stride_px
        self.inference_parameters = inference_parameters
        self.px_in_rlayer_units = px_in_rlayer_units

        self.extent = self._calculate_extent(file_extent)  # type: QgsRectangle  # tile extent in CRS cordinates

    def _calculate_extent(self, file_extent):
        tile_extent = QgsRectangle(file_extent)  # copy
        x_min = file_extent.xMinimum() + self.start_pixel_x * self.px_in_rlayer_units
        y_min = file_extent.yMinimum() + self.start_pixel_y * self.px_in_rlayer_units
        tile_extent.setXMinimum(x_min)
        # extent needs to be on the further edge (so including the corner pixel, hence we do not subtract 1)
        tile_extent.setXMaximum(x_min + self.inference_parameters.tile_size_px * self.px_in_rlayer_units)
        tile_extent.setYMinimum(y_min)
        tile_extent.setYMaximum(y_min + self.inference_parameters.tile_size_px * self.px_in_rlayer_units)
        return tile_extent

    def get_slice_on_full_image_for_copying(self):
        """
        As we are doing processing with overlap, we are not going to copy the entire tile result to final image,
        but only the part that is not overlapping with the neighbouring tiles.
        Edge tiles have special handling too.

        :return Slice to be used on the full image
        """
        half_overlap = (self.inference_parameters.tile_size_px - self.stride_px) // 2

        # 'core' part of the tile (not overlapping with other tiles), for sure copied for each tile
        x_min = self.start_pixel_x + half_overlap
        x_max = self.start_pixel_x + self.inference_parameters.tile_size_px - half_overlap - 1
        y_min = self.start_pixel_y + half_overlap
        y_max = self.start_pixel_y + self.inference_parameters.tile_size_px - half_overlap - 1

        # edge tiles handling
        if self.x_bin_number == 0:
            x_min -= half_overlap
        if self.y_bin_number == 0:
            y_min -= half_overlap
        if self.x_bin_number == self.x_bins_number-1:
            x_max += half_overlap
        if self.y_bin_number == self.y_bins_number-1:
            y_max += half_overlap

        roi_slice = np.s_[y_min:y_max + 1, x_min:x_max + 1]
        return roi_slice

    def get_slice_on_tile_image_for_copying(self, roi_slice_on_full_image = None):
        """
        Similar to _get_slice_on_full_image_for_copying, but ROI is a slice on the tile
        """
        if not roi_slice_on_full_image:
            roi_slice_on_full_image = self.get_slice_on_full_image_for_copying()

        r = roi_slice_on_full_image
        roi_slice_on_tile = np.s_[
                            r[0].start - self.start_pixel_y : r[0].stop - self.start_pixel_y,
                            r[1].start - self.start_pixel_x : r[1].stop - self.start_pixel_x
                            ]
        return roi_slice_on_tile


class MapProcessor(QgsTask):
    finished_signal = pyqtSignal(str)  # error message if finished with error, empty string otherwise
    show_img_signal = pyqtSignal(object)  # request to show an image

    def __init__(self,
                 rlayer: QgsRasterLayer,
                 processed_extent: QgsRectangle,
                 inference_parameters: InferenceParameters):
        print(processed_extent)
        QgsTask.__init__(self, self.__class__.__name__)
        self.rlayer = rlayer
        self.processed_extent = processed_extent
        self.inference_parameters = inference_parameters

        self.stride_px = self.inference_parameters.processing_stride_px
        self.px_in_rlayer_units = self.convert_meters_to_rlayer_units(
            self.rlayer, self.inference_parameters.resolution_m_per_px)  # number of rlayer units for one tile pixel

        # processed rlayer dimensions
        self.img_size_x_pixels = round(self.processed_extent.width() / self.px_in_rlayer_units)
        self.img_size_y_pixels = round(self.processed_extent.height() / self.px_in_rlayer_units)

        self.x_bins_number = (self.img_size_x_pixels - self.inference_parameters.tile_size_px) // self.stride_px + 1  # use int casting instead of // to have always at least 1
        self.y_bins_number = (self.img_size_y_pixels - self.inference_parameters.tile_size_px) // self.stride_px + 1

    def run(self):
        print('run...')
        return self._process()

    @staticmethod
    def convert_meters_to_rlayer_units(rlayer, distance_m) -> float:
        """ How many map units are there in one meter """
        # TODO - potentially implement conversions from other units
        if rlayer.crs().mapUnits() != QgsUnitTypes.DistanceUnit.DistanceMeters:
            raise Exception("Unsupported layer units")
        return distance_m

    def finished(self, result):
        print(f'finished. Res: {result = }')
        if result:
            self.finished_signal.emit('')
        else:
            self.finished_signal.emit('Processing error')

    def is_busy(self):
        return True

    def _get_image(self, rlayer, extent, inference_parameters: InferenceParameters) -> np.ndarray:
        expected_meters_per_pixel = inference_parameters.resolution_cm_per_px / 100
        expected_units_per_pixel = self.convert_meters_to_rlayer_units(rlayer, expected_meters_per_pixel)
        expected_units_per_pixel_2d = expected_units_per_pixel, expected_units_per_pixel
        # to get all pixels - use the 'rlayer.rasterUnitsPerPixelX()' instead of 'expected_units_per_pixel_2d'
        image_size = round((extent.width()) / expected_units_per_pixel_2d[0]), \
                     round((extent.height()) / expected_units_per_pixel_2d[1])

        # sanity check, that we gave proper extent as parameter
        assert image_size[0] == inference_parameters.tile_size_px
        assert image_size[1] == inference_parameters.tile_size_px

        band_count = rlayer.bandCount()
        band_data = []

        # enable resampling
        data_provider = rlayer.dataProvider()
        data_provider.enableProviderResampling(True)
        original_resampling_method = data_provider.zoomedInResamplingMethod()
        data_provider.setZoomedInResamplingMethod(data_provider.ResamplingMethod.Bilinear)
        data_provider.setZoomedOutResamplingMethod(data_provider.ResamplingMethod.Bilinear)

        for band_number in range(1, band_count + 1):
            raster_block = rlayer.dataProvider().block(
                band_number,
                extent,
                image_size[0], image_size[1])
            rb = raster_block
            rb.height(), rb.width()
            raw_data = rb.data()
            bytes_array = bytes(raw_data)
            dt = rb.dataType()
            if dt == dt.__class__.Byte:
                number_of_channels = 1
            elif dt == dt.__class__.ARGB32:
                number_of_channels = 4
            else:
                raise Exception("Invalid type!")

            a = np.frombuffer(bytes_array, dtype=np.uint8)
            b = a.reshape((image_size[1], image_size[0], number_of_channels))
            band_data.append(b)

        data_provider.setZoomedInResamplingMethod(original_resampling_method)  # restore old resampling method

        if band_count == 4:
            band_data = [band_data[2], band_data[1], band_data[0], band_data[3]]

        img = np.concatenate(band_data, axis=2)
        return img

    def _process(self):
        total_tiles = self.x_bins_number * self.y_bins_number
        final_shape_px = (self.img_size_y_pixels, self.img_size_x_pixels)
        full_predicted_img = np.zeros(final_shape_px, np.uint8)

        if total_tiles < 1:
            raise Exception("TODO! Add support for partial tiles!")
        # TODO - add support for to small images - padding for the last bin
        # (and also bins_number calculation, to have at least one)

        # TODO - add processing in background thread
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
                                         file_extent=self.processed_extent,
                                         px_in_rlayer_units=self.px_in_rlayer_units)
                tile_img = self._get_image(self.rlayer, tile_params.extent, self.inference_parameters)
                self.show_img_signal.emit(tile_img)
                # tile_result = self._process_tile(tile_img)
                # self.show_img_signal.emit(tile_output)
                # self._set_mask_on_full_img(tile_result=tile_result,
                #                            full_predicted_img=full_predicted_img,
                #                            tile_params=tile_params)

        # self.show_img_signal.emit(full_predicted_img)
        return True

    def _set_mask_on_full_img(self, full_predicted_img, tile_result, tile_params: TileParams):
        roi_slice_on_full_image = tile_params.get_slice_on_full_image_for_copying()
        roi_slice_on_tile_image = tile_params.get_slice_on_tile_image_for_copying(roi_slice_on_full_image)
        full_predicted_img[roi_slice_on_full_image] = tile_result[roi_slice_on_tile_image]

    def _process_tile(self, tile_img: np.ndarray) -> np.ndarray:
        # TODO
        tile_img = copy.copy(tile_img)
        tile_img = tile_img[:, :, 1]
        tile_img[tile_img < 100] = 0
        tile_img[tile_img > 100] = 255
        result = tile_img
        return result

