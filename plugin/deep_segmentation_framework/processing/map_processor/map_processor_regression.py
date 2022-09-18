import tempfile
from typing import List
import os

import cv2
import numpy as np
from qgis.core import QgsCoordinateReferenceSystem
from qgis.core import QgsRasterLayer
from osgeo import gdal, osr, ogr

from qgis.core import QgsVectorLayer
from qgis.core import QgsProject

from deep_segmentation_framework.common.processing_parameters.map_processing_parameters import ModelOutputFormat
from deep_segmentation_framework.common.processing_parameters.regression_parameters import RegressionParameters
from deep_segmentation_framework.processing import processing_utils
from deep_segmentation_framework.common.defines import IS_DEBUG
from deep_segmentation_framework.processing.map_processor.map_processing_result import MapProcessingResult, \
    MapProcessingResultCanceled, MapProcessingResultSuccess
from deep_segmentation_framework.processing.map_processor.map_processor import MapProcessor
from deep_segmentation_framework.processing.map_processor.map_processor_with_model import MapProcessorWithModel

if IS_DEBUG:
    pass


class MapProcessorRegression(MapProcessorWithModel):
    def __init__(self,
                 params: RegressionParameters,
                 **kwargs):
        super().__init__(
            params=params,
            model=params.model,
            **kwargs)
        self.regression_parameters = params
        self.model = params.model
        self._result_imgs = None

    def get_result_imgs(self):
        return self._result_imgs

    def _run(self) -> MapProcessingResult:
        number_of_output_channels = len(self._get_indexes_of_model_output_channels_to_create())
        final_shape_px = (self.img_size_y_pixels, self.img_size_x_pixels)
        full_result_imgs = [np.zeros(final_shape_px, np.uint8) for i in range(number_of_output_channels)]

        for tile_img, tile_params in self.tiles_generator():
            if self.isCanceled():
                return MapProcessingResultCanceled()

            tile_results = self._process_tile(tile_img)
            for i in range(number_of_output_channels):
                tile_params.set_mask_on_full_img(
                    tile_result=tile_results[i],
                    full_result_img=full_result_imgs[i])

        # plt.figure(); plt.imshow(full_result_img); plt.show(block=False); plt.pause(0.001)
        full_result_imgs = self.limit_extended_extent_images_to_base_extent_with_mask(full_imgs=full_result_imgs)
        self._result_imgs = full_result_imgs
        self._create_rlayers_from_images_for_base_extent(self._result_imgs)

        result_message = self._create_result_message(self._result_imgs)
        return MapProcessingResultSuccess(result_message)

    def _create_result_message(self, result_imgs: List[np.ndarray]) -> str:
        channels = self._get_indexes_of_model_output_channels_to_create()
        txt = f'Regression done for {len(channels)} model output channels, with the following statistics:\n'
        for i, channel_id in enumerate(channels):
            result_img = result_imgs[i]
            average_value = np.mean(result_img)
            std = np.std(result_img)
            txt += f' - class {channel_id}: average_value = {average_value:.2f} (std = {std:.2f})\n'

        if len(channels) > 0:
            total_area = result_img.shape[0] * result_img.shape[1] * self.params.resolution_m_per_px**2
            txt += f'Total are is {total_area} m^2'
        return txt

    def limit_extended_extent_images_to_base_extent_with_mask(self, full_imgs: List[np.ndarray]):
        """
        Same as 'limit_extended_extent_image_to_base_extent_with_mask' but for a list of images.
        See `limit_extended_extent_image_to_base_extent_with_mask` for details.
        :param full_imgs:
        :return:
        """
        result_imgs = []
        for i in range(len(full_imgs)):
            result_img = self.limit_extended_extent_image_to_base_extent_with_mask(full_img=full_imgs[i])
            result_imgs.append(result_img)

        return result_imgs

    def load_rlayer_from_file(self, file_path):
        """
        Create raster layer from tif file
        """
        rlayer = QgsRasterLayer(file_path, os.path.basename(file_path))
        if rlayer.width() == 0:
            raise Exception("0 width - rlayer not loaded properly. Probably invalid file path?")
        rlayer.setCrs(self.rlayer.crs())
        return rlayer

    def _create_rlayers_from_images_for_base_extent(self, result_imgs: List[np.ndarray]):
        group = QgsProject.instance().layerTreeRoot().insertGroup(0, 'model_output')

        # TODO: We are creating a new file for each layer.
        # Maybe can we pass ownership of this file to QGis?
        # Or maybe even create vlayer directly from array, without a file?

        tmp_dir = tempfile.TemporaryDirectory()
        tmp_dir_path = os.path.join(tmp_dir.name, 'qgis')

        for i, channel_id in enumerate(self._get_indexes_of_model_output_channels_to_create()):
            result_img = result_imgs[i]
            result_img *= 255
            result_img = np.clip(result_img, 0, 255)
            result_img = result_img.astype(np.uint8)

            file_path = os.path.join(tmp_dir_path, f'channel_{channel_id}.tif')
            self.save_result_img_as_tif(file_path=file_path, img=result_img)

            rlayer = self.load_rlayer_from_file(file_path)
            # TODO set color mapping and transparency
            # prov = vlayer.dataProvider()
            # color = rlayer.renderer().symbol().color()
            # OUTPUT_VLAYER_COLOR_TRANSPARENCY = 80
            # color.setAlpha(OUTPUT_VLAYER_COLOR_TRANSPARENCY)
            # rlayer.renderer().symbol().setColor(color)

            QgsProject.instance().addMapLayer(rlayer, False)
            group.addLayer(rlayer)

    def save_result_img_as_tif(self, file_path: str, img: np.ndarray):
        # def getGeoTransform(extent_minmax, nlines, ncols):
        #     resx = (extent_minmax[2] - extent_minmax[0]) / ncols
        #     resy = (extent_minmax[3] - extent_minmax[1]) / nlines
        #     return [extent[0], resx, 0, extent[3], 0, -resy]

        data = img
        extent = self.base_extent
        crs = self.rlayer.crs()

        geo_transform = [extent.xMinimum(), self.rlayer_units_per_pixel, 0,
                         extent.yMinimum(), 0, -self.rlayer_units_per_pixel]


        driver = gdal.GetDriverByName('GTiff')
        nlines = data.shape[0]
        ncols = data.shape[1]
        data_type = gdal.GDT_Byte
        grid_data = driver.Create('grid_data', ncols, nlines, 1, data_type)  # , options)
        grid_data.GetRasterBand(1).WriteArray(data)

        srs = osr.SpatialReference()
        srs.ImportFromProj4('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
        # srs.ImportFromEPSG()

        grid_data.SetProjection(srs.ExportToWkt())
        # grid_data.SetGeoTransform(getGeoTransform(extent_minmax, nlines, ncols))
        grid_data.SetGeoTransform(geo_transform)
        driver.CreateCopy(file_path, grid_data, 0)
        print(f'***** {file_path = }')

    def _process_tile(self, tile_img: np.ndarray) -> np.ndarray:
        result = self.model.process(tile_img)
        result *= self.regression_parameters.output_scaling
        return result

