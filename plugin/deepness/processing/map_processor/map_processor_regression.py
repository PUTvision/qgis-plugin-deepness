import tempfile
import uuid
from typing import List
import os

import cv2
import numpy as np
from qgis.core import QgsCoordinateReferenceSystem
from qgis.core import QgsRasterLayer
from osgeo import gdal, osr, ogr

from qgis.core import QgsVectorLayer
from qgis.core import QgsProject

from deepness.common.misc import TMP_DIR_PATH
from deepness.common.processing_parameters.map_processing_parameters import ModelOutputFormat
from deepness.common.processing_parameters.regression_parameters import RegressionParameters
from deepness.processing import processing_utils
from deepness.common.defines import IS_DEBUG
from deepness.processing.map_processor.map_processing_result import MapProcessingResult, \
    MapProcessingResultCanceled, MapProcessingResultSuccess
from deepness.processing.map_processor.map_processor import MapProcessor
from deepness.processing.map_processor.map_processor_with_model import MapProcessorWithModel


class MapProcessorRegression(MapProcessorWithModel):
    """
    MapProcessor specialized for Regression model (where each pixel has a value representing some feature intensity)
    """

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

        # NOTE: consider whether we can use float16/uint16 as datatype
        full_result_imgs = [np.zeros(final_shape_px, np.float32) for i in range(number_of_output_channels)]

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
            txt += f' - {self.model.get_channel_name(channel_id)}: average_value = {average_value:.2f} (std = {std:.2f}, ' \
                   f'min={np.min(result_img)}, max={np.max(result_img)})\n'

        if len(channels) > 0:
            total_area = result_img.shape[0] * result_img.shape[1] * self.params.resolution_m_per_px**2
            txt += f'Total are is {total_area:.2f} m^2'
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
        file_name = os.path.basename(file_path)
        base_file_name = file_name.split('___')[0]  # we remove the random_id string we created a moment ago
        rlayer = QgsRasterLayer(file_path, base_file_name)
        if rlayer.width() == 0:
            raise Exception("0 width - rlayer not loaded properly. Probably invalid file path?")
        rlayer.setCrs(self.rlayer.crs())
        return rlayer

    def _create_rlayers_from_images_for_base_extent(self, result_imgs: List[np.ndarray]):
        group = QgsProject.instance().layerTreeRoot().insertGroup(0, 'model_output')

        # TODO: We are creating a new file for each layer.
        # Maybe can we pass ownership of this file to QGis?
        # Or maybe even create vlayer directly from array, without a file?

        for i, channel_id in enumerate(self._get_indexes_of_model_output_channels_to_create()):
            result_img = result_imgs[i]
            random_id = str(uuid.uuid4()).replace('-', '')
            file_path = os.path.join(TMP_DIR_PATH, f'{self.model.get_channel_name(channel_id)}___{random_id}.tif')
            self.save_result_img_as_tif(file_path=file_path, img=result_img)

            rlayer = self.load_rlayer_from_file(file_path)
            OUTPUT_RLAYER_OPACITY = 0.5
            rlayer.renderer().setOpacity(OUTPUT_RLAYER_OPACITY)

            QgsProject.instance().addMapLayer(rlayer, False)
            group.addLayer(rlayer)

    def save_result_img_as_tif(self, file_path: str, img: np.ndarray):
        """
        As we cannot pass easily an numpy array to be displayed as raster layer, we create temporary geotif files,
        which will be loaded as layer later on

        Partially based on example from:
        https://gis.stackexchange.com/questions/82031/gdal-python-set-projection-of-a-raster-not-working
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        extent = self.base_extent
        crs = self.rlayer.crs()

        geo_transform = [extent.xMinimum(), self.rlayer_units_per_pixel, 0,
                         extent.yMaximum(), 0, -self.rlayer_units_per_pixel]

        driver = gdal.GetDriverByName('GTiff')
        n_lines = img.shape[0]
        n_cols = img.shape[1]
        # data_type = gdal.GDT_Byte
        data_type = gdal.GDT_Float32
        grid_data = driver.Create('grid_data', n_cols, n_lines, 1, data_type)  # , options)
        grid_data.GetRasterBand(1).WriteArray(img)

        # crs().srsid()  - maybe we can use the ID directly - but how?
        # srs.ImportFromEPSG()
        srs = osr.SpatialReference()
        srs.SetFromUserInput(crs.authid())

        grid_data.SetProjection(srs.ExportToWkt())
        grid_data.SetGeoTransform(geo_transform)
        driver.CreateCopy(file_path, grid_data, 0)
        print(f'***** {file_path = }')

    def _process_tile(self, tile_img: np.ndarray) -> np.ndarray:
        result = self.model.process(tile_img)
        result[np.isnan(result)] = 0
        result *= self.regression_parameters.output_scaling

        # NOTE - currently we are saving result as float32, so we are losing some accuraccy.
        # result = np.clip(result, 0, 255)  # old version with uint8_t - not used anymore
        result = result.astype(np.float32)

        return result

