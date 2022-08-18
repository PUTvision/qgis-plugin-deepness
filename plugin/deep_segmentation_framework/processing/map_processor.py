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


class MapProcessor(QgsTask):
    finished_signal = pyqtSignal(str)  # error message if finished with error, empty string otherwise
    show_img_signal = pyqtSignal(object)  # request to show an image

    def __init__(self,
                 rlayer: QgsRasterLayer,
                 processed_extent: QgsRectangle,
                 inference_parameters: InferenceParameters):
        QgsTask.__init__(self, self.__class__.__name__)
        self.rlayer = rlayer
        self.processed_extent = processed_extent
        self.inference_parameters = inference_parameters

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

    def _show_raster_as_image(self, rlayer, extent, inference_parameters: InferenceParameters):
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
            dt
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
        self.show_img_signal.emit(img)

    def _process(self):
        stride = self.inference_parameters.tile_size_px - self.inference_parameters.processing_overlap_px
        px_in_rlayer_units = self.convert_meters_to_rlayer_units(
            self.rlayer, self.inference_parameters.resolution_m_per_px)  # number of rlayer units for one tile pixel
        img_size_x_pixels = round(self.processed_extent.width() / px_in_rlayer_units)
        img_size_y_pixels = round(self.processed_extent.height() / px_in_rlayer_units)

        x_bins_number = (img_size_x_pixels - self.inference_parameters.tile_size_px) // stride + 1  # use int casting instead of // to have always at least 1
        y_bins_number = (img_size_y_pixels - self.inference_parameters.tile_size_px) // stride + 1
        total_tiles = x_bins_number * y_bins_number

        final_shape_px = (img_size_x_pixels, img_size_y_pixels)
        full_predicted_img = np.zeros(final_shape_px, np.uint8)

        if total_tiles < 1:
            raise Exception("TODO! Add support for partial tiles!")
        # TODO - add support for to small images - padding for the last bin
        # (and also bins_number calculation, to have at least one)

        # TODO - add processing in background thread
        for y_bin_number in range(y_bins_number):
            for x_bin_number in range(x_bins_number):
                if self.isCanceled():
                    return False
                tile_no = y_bin_number * x_bins_number + x_bin_number
                progress = int(tile_no / total_tiles * 100)
                self.setProgress(progress)
                print(f" Processing tile {tile_no} / {total_tiles} [{progress:.2f}%]")

                start_pixel_x = x_bin_number * stride
                start_pixel_y = y_bin_number * stride

                tile_extent = QgsRectangle(self.processed_extent)  # copy
                x_min = self.processed_extent.xMinimum() + start_pixel_x * px_in_rlayer_units
                y_min = self.processed_extent.yMinimum() + start_pixel_y * px_in_rlayer_units
                tile_extent.setXMinimum(x_min)
                # extent needs to be on the further edge (so including the corner pixel, hence we do not subtract 1)
                tile_extent.setXMaximum(x_min + self.inference_parameters.tile_size_px * px_in_rlayer_units)
                tile_extent.setYMinimum(y_min)
                tile_extent.setYMaximum(y_min + self.inference_parameters.tile_size_px * px_in_rlayer_units)
                self._show_raster_as_image(self.rlayer, tile_extent, self.inference_parameters)
        return True

    def finished(self, result):
        print('finished...')
        print('finished2...')
        if result:
            self.finished_signal.emit('')
        else:
            self.finished_signal.emit('Processing error')

    def is_busy(self):
        return True
