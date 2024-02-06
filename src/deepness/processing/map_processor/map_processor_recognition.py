""" This file implements map processing for Recognition model """

import os
import uuid
from typing import List

import numpy as np
from deepness.common.lazy_package_loader import LazyPackageLoader
from deepness.common.misc import TMP_DIR_PATH
from deepness.common.processing_parameters.recognition_parameters import \
    RecognitionParameters
from deepness.processing.map_processor.map_processing_result import (
    MapProcessingResult, MapProcessingResultCanceled,
    MapProcessingResultSuccess)
from deepness.processing.map_processor.map_processor_with_model import \
    MapProcessorWithModel
from numpy.linalg import norm
from osgeo import gdal, osr
from qgis.core import QgsProject, QgsRasterLayer

cv2 = LazyPackageLoader('cv2')


class MapProcessorRecognition(MapProcessorWithModel):
    """
    MapProcessor specialized for Recognition model
    """

    def __init__(self, params: RecognitionParameters, **kwargs):
        super().__init__(params=params, model=params.model, **kwargs)
        self.recognition_parameters = params
        self.model = params.model
        self._result_imgs = None

    def get_result_imgs(self):
        return self._result_imgs

    def _run(self) -> MapProcessingResult:
        try:
            print("*" * 80)
            print(self.recognition_parameters.query_image_path)
            query_img = cv2.imread(self.recognition_parameters.query_image_path)
        except Exception as err:
            print(err)
            raise RuntimeError("unable to open image")
            
        query_img_emb = self.model.process(query_img)[0]
        
        final_shape_px = (
            self.img_size_y_pixels,
            self.img_size_x_pixels,
        )
        
        stride = self.stride_px
        full_result_img = np.zeros(final_shape_px, np.float32)
        mask = np.zeros_like(full_result_img, dtype=np.int16)
        highest = 0
        for tile_img, tile_params in self.tiles_generator():
            if self.isCanceled():
                return MapProcessingResultCanceled()

            # See note in the class description why are we adding/subtracting 1 here
            tile_result = self._process_tile(tile_img)[0]
            
            # cosine similarity
            cossim = np.dot(query_img_emb[0],tile_result[0])/(norm(query_img_emb[0])*norm(tile_result[0]))
            
            x_bin = tile_params.x_bin_number
            y_bin = tile_params.y_bin_number
            size = self.params.tile_size_px
            if cossim > highest:
                highest = cossim
                x_high = x_bin
                y_high = y_bin
            full_result_img[y_bin*stride:y_bin*stride+size, x_bin*stride:x_bin*stride +size] +=  cossim
            mask[y_bin*stride:y_bin*stride+size, x_bin*stride:x_bin*stride +size] += 1

        full_result_img = full_result_img/mask
        self._result_img = full_result_img

        self._create_rlayers_from_images_for_base_extent(self._result_img, x_high, y_high, size, stride)#*255).astype(int))
        result_message = self._create_result_message(self._result_img, x_high*self.params.tile_size_px, y_high*self.params.tile_size_px)
        return MapProcessingResultSuccess(result_message)

    def _create_result_message(self, result_img: List[np.ndarray], x_high, y_high) -> str:
        txt = f"Recognition ended, best result found at {x_high}, {y_high}, {result_img.shape}"
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
        # full_img = cv2.copyTo(src=full_img, mask=self.area_mask_img)

        b = self.base_extent_bbox_in_full_image
        result_img = full_img[
            int(b.y_min * self.recognition_parameters.scale_factor) : int(
                b.y_max * self.recognition_parameters.scale_factor
            ),
            int(b.x_min * self.recognition_parameters.scale_factor) : int(
                b.x_max * self.recognition_parameters.scale_factor
            ),
            :,
        ]
        return result_img

    def load_rlayer_from_file(self, file_path):
        """
        Create raster layer from tif file
        """
        file_name = os.path.basename(file_path)
        base_file_name = file_name.split("___")[
            0
        ]  # we remove the random_id string we created a moment ago
        rlayer = QgsRasterLayer(file_path, base_file_name)
        if rlayer.width() == 0:
            raise Exception(
                "0 width - rlayer not loaded properly. Probably invalid file path?"
            )
        rlayer.setCrs(self.rlayer.crs())
        return rlayer

    def _create_rlayers_from_images_for_base_extent(
        self, result_img: np.ndarray,
        x_high,
        y_high,
        size,
        stride
    ):
        group = (
            QgsProject.instance()
            .layerTreeRoot()
            .insertGroup(0, "Cosine similarity score")
        )
        
        min_value = np.min(result_img)
        y = y_high * stride
        x = x_high * stride
        print(f"{x},{y}")
        
        result_img[y, x:x+size-1] = 1
        result_img[y+size-1, x:x+size-1] = 1
        result_img[y:y+size-1, x] = 1
        result_img[y:y+size-1, x+size-1] = 1

        # TODO: We are creating a new file for each layer.
        # Maybe can we pass ownership of this file to QGis?
        # Or maybe even create vlayer directly from array, without a file?

   #     for i, channel_id in enumerate(["Super Resolution"]):
        random_id = str(uuid.uuid4()).replace("-", "")
        file_path = os.path.join(TMP_DIR_PATH, f"{random_id}.tif")
        self.save_result_img_as_tif(file_path=file_path, img=np.expand_dims(result_img, axis=2))

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

        geo_transform = [
            extent.xMinimum(),
            self.rlayer_units_per_pixel, # / self.recognition_parameters.scale_factor,
            0,
            extent.yMaximum(),
            0,
            -self.rlayer_units_per_pixel, # / self.recognition_parameters.scale_factor,
        ]

        driver = gdal.GetDriverByName("GTiff")
        n_lines = img.shape[0]
        n_cols = img.shape[1]
        n_chanels = img.shape[2]
        # data_type = gdal.GDT_Byte
        data_type = gdal.GDT_Float32
        grid_data = driver.Create(
            "grid_data", n_cols, n_lines, n_chanels, data_type
        )  # , options)
        # loop over chanels
        for i in range(1, img.shape[2] + 1):
            grid_data.GetRasterBand(i).WriteArray(img[:, :, i - 1])

        # crs().srsid()  - maybe we can use the ID directly - but how?
        # srs.ImportFromEPSG()
        srs = osr.SpatialReference()
        srs.SetFromUserInput(crs.authid())

        grid_data.SetProjection(srs.ExportToWkt())
        grid_data.SetGeoTransform(geo_transform)
        driver.CreateCopy(file_path, grid_data, 0)

    def _process_tile(self, tile_img: np.ndarray) -> np.ndarray:
        result = self.model.process(tile_img)
     #   result[np.isnan(result)] = 0

        # NOTE - currently we are saving result as float32, so we are losing some accuraccy.
        # result = np.clip(result, 0, 255)  # old version with uint8_t - not used anymore
     #   result = result.astype(np.float32)

        return result
