""" This file implements map processing for Super Resolution model """

import uuid
from typing import List
import os
import uuid
from typing import List

import numpy as np
from osgeo import gdal, osr
from qgis.core import QgsProject
from qgis.core import QgsRasterLayer

from deepness.common.misc import TMP_DIR_PATH
from deepness.common.processing_parameters.superresolution_parameters import SuperresolutionParameters
from deepness.processing.map_processor.map_processing_result import MapProcessingResult, \
    MapProcessingResultCanceled, MapProcessingResultSuccess
from deepness.processing.map_processor.map_processor_with_model import MapProcessorWithModel


class MapProcessorSuperresolution(MapProcessorWithModel):
    """
    MapProcessor specialized for Super Resolution model (whic is is used to upscale the input image to a higher resolution)
    """

    def __init__(self,
                 params: SuperresolutionParameters,
                 **kwargs):
        super().__init__(
            params=params,
            model=params.model,
            **kwargs)
        self.superresolution_parameters = params
        self.model = params.model
        self._result_imgs = None

    def get_result_imgs(self):
        return self._result_imgs

    def _run(self) -> MapProcessingResult:
        number_of_output_channels = len(self._get_indexes_of_model_output_channels_to_create())
        final_shape_px = (int(self.img_size_y_pixels*self.superresolution_parameters.scale_factor), int(self.img_size_x_pixels*self.superresolution_parameters.scale_factor), number_of_output_channels)

        # NOTE: consider whether we can use float16/uint16 as datatype
        full_result_imgs = np.zeros(final_shape_px, np.float32) 

        for tile_img, tile_params in self.tiles_generator():
            if self.isCanceled():
                return MapProcessingResultCanceled()

            tile_results = self._process_tile(tile_img)
            full_result_imgs[int(tile_params.start_pixel_y*self.superresolution_parameters.scale_factor):int((tile_params.start_pixel_y+tile_params.stride_px)*self.superresolution_parameters.scale_factor),
                                int(tile_params.start_pixel_x*self.superresolution_parameters.scale_factor):int((tile_params.start_pixel_x+tile_params.stride_px)*self.superresolution_parameters.scale_factor),
                                :] = tile_results.transpose(1, 2, 0)  # transpose to chanels last

        # plt.figure(); plt.imshow(full_result_img); plt.show(block=False); plt.pause(0.001)
        full_result_imgs = self.limit_extended_extent_image_to_base_extent_with_mask(full_img=full_result_imgs )
        self._result_imgs = full_result_imgs
        self._create_rlayers_from_images_for_base_extent(self._result_imgs)

        result_message = self._create_result_message(self._result_imgs)
        return MapProcessingResultSuccess(result_message)

    def _create_result_message(self, result_img: List[np.ndarray]) -> str:
        channels = self._get_indexes_of_model_output_channels_to_create()
        txt = f'Super-resolution done \n'

        if len(channels) > 0:
            total_area = result_img.shape[0] * result_img.shape[1] * (self.params.resolution_m_per_px /self.superresolution_parameters.scale_factor)**2
            txt += f'Total are is {total_area:.2f} m^2'
        return txt
    def limit_extended_extent_image_to_base_extent_with_mask(self, full_img):
        """
        Limit an image which is for extended_extent to the base_extent image.
        If a limiting polygon was used for processing, it will be also applied.
        :param full_img:
        :return:
        """
        # TODO look for some inplace operation to save memory
        # cv2.copyTo(src=full_img, mask=area_mask_img, dst=full_img)  # this doesn't work due to implementation details
        #full_img = cv2.copyTo(src=full_img, mask=self.area_mask_img)

        b = self.base_extent_bbox_in_full_image
        result_img = full_img[int(b.y_min*self.superresolution_parameters.scale_factor):int(b.y_max*self.superresolution_parameters.scale_factor),
                                int(b.x_min*self.superresolution_parameters.scale_factor):int(b.x_max*self.superresolution_parameters.scale_factor),
                                :]
        return result_img

    

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
        group = QgsProject.instance().layerTreeRoot().insertGroup(0, 'Super Resolution Results')

        # TODO: We are creating a new file for each layer.
        # Maybe can we pass ownership of this file to QGis?
        # Or maybe even create vlayer directly from array, without a file?

        for i, channel_id in enumerate(['Super Resolution']):
            result_img = result_imgs
            random_id = str(uuid.uuid4()).replace('-', '')
            file_path = os.path.join(TMP_DIR_PATH, f'{channel_id}___{random_id}.tif')
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

        geo_transform = [extent.xMinimum(), self.rlayer_units_per_pixel/self.superresolution_parameters.scale_factor, 0,
                         extent.yMaximum(), 0, -self.rlayer_units_per_pixel/self.superresolution_parameters.scale_factor]

        driver = gdal.GetDriverByName('GTiff')
        n_lines = img.shape[0]
        n_cols = img.shape[1]
        n_chanels = img.shape[2]
        # data_type = gdal.GDT_Byte
        data_type = gdal.GDT_Float32
        grid_data = driver.Create('grid_data', n_cols, n_lines, n_chanels, data_type)  # , options)
        #loop over chanels
        for i in range(1, img.shape[2]+1):
            grid_data.GetRasterBand(i).WriteArray(img[:, :, i-1]) 

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
        result *= self.superresolution_parameters.output_scaling

        # NOTE - currently we are saving result as float32, so we are losing some accuraccy.
        # result = np.clip(result, 0, 255)  # old version with uint8_t - not used anymore
        result = result.astype(np.float32)

        return result
