""" This file implements map processing for regression model """

import os
import uuid
from typing import List

import numpy as np
from osgeo import gdal, osr
from qgis.core import QgsProject, QgsRasterLayer

from deepness.common.misc import TMP_DIR_PATH
from deepness.common.processing_parameters.regression_parameters import RegressionParameters
from deepness.processing.map_processor.map_processing_result import (MapProcessingResult, MapProcessingResultCanceled,
                                                                     MapProcessingResultSuccess)
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

    def _run(self) -> MapProcessingResult:
        number_of_output_channels = len(self._get_indexes_of_model_output_channels_to_create())
        final_shape_px = (number_of_output_channels, self.img_size_y_pixels, self.img_size_x_pixels)

        # NOTE: consider whether we can use float16/uint16 as datatype
        full_result_imgs = self._get_array_or_mmapped_array(final_shape_px)

        for tile_img_batched, tile_params_batched in self.tiles_generator_batched():
            if self.isCanceled():
                return MapProcessingResultCanceled()

            tile_results_batched = self._process_tile(tile_img_batched)

            for tile_results, tile_params in zip(tile_results_batched, tile_params_batched):
                tile_params.set_mask_on_full_img(
                    tile_result=tile_results,
                    full_result_img=full_result_imgs)

        # plt.figure(); plt.imshow(full_result_img); plt.show(block=False); plt.pause(0.001)
        full_result_imgs = self.limit_extended_extent_images_to_base_extent_with_mask(full_imgs=full_result_imgs)
        self.set_results_img(full_result_imgs)

        gui_delegate = self._create_rlayers_from_images_for_base_extent(self.get_result_img())
        result_message = self._create_result_message(self.get_result_img())
        return MapProcessingResultSuccess(
            message=result_message,
            gui_delegate=gui_delegate,
        )

    def _create_result_message(self, result_imgs: List[np.ndarray]) -> str:
        txt = f'Regression done, with the following statistics:\n'
        for output_id, _ in enumerate(self._get_indexes_of_model_output_channels_to_create()):
            result_img = result_imgs[output_id]
            
            average_value = np.mean(result_img)
            std = np.std(result_img)
            
            txt += f' - {self.model.get_channel_name(output_id, 0)}: average_value = {average_value:.2f} (std = {std:.2f}, ' \
                   f'min={np.min(result_img)}, max={np.max(result_img)})\n'

        return txt

    def limit_extended_extent_images_to_base_extent_with_mask(self, full_imgs: List[np.ndarray]):
        """
        Same as 'limit_extended_extent_image_to_base_extent_with_mask' but for a list of images.
        See `limit_extended_extent_image_to_base_extent_with_mask` for details.
        :param full_imgs:
        :return:
        """
        return self.limit_extended_extent_image_to_base_extent_with_mask(full_img=full_imgs)

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
        # TODO: We are creating a new file for each layer.
        # Maybe can we pass ownership of this file to QGis?
        # Or maybe even create vlayer directly from array, without a file?
        rlayers = []

        for output_id, _ in enumerate(self._get_indexes_of_model_output_channels_to_create()):
            file_path = os.path.join(TMP_DIR_PATH, f'{self.model.get_channel_name(output_id, 0)}.tif')
            self.save_result_img_as_tif(file_path=file_path, img=result_imgs[output_id])

            rlayer = self.load_rlayer_from_file(file_path)
            OUTPUT_RLAYER_OPACITY = 0.5
            rlayer.renderer().setOpacity(OUTPUT_RLAYER_OPACITY)
            rlayers.append(rlayer)

        def add_to_gui():
            group = QgsProject.instance().layerTreeRoot().insertGroup(0, 'model_output')
            for rlayer in rlayers:
                QgsProject.instance().addMapLayer(rlayer, False)
                group.addLayer(rlayer)

        return add_to_gui

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
        many_result = self.model.process(tile_img)
        many_outputs = []

        for result in many_result:
            result[np.isnan(result)] = 0
            result *= self.regression_parameters.output_scaling

            # NOTE - currently we are saving result as float32, so we are losing some accuraccy.
            # result = np.clip(result, 0, 255)  # old version with uint8_t - not used anymore
            result = result.astype(np.float32)

            if len(result.shape) == 3:
                result = np.expand_dims(result, axis=1)

            many_outputs.append(result[:, 0])

        many_outputs = np.array(many_outputs).transpose((1, 0, 2, 3))

        return many_outputs
