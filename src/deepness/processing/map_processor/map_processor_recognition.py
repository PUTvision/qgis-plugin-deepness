""" This file implements map processing for Recognition model """

import os
import uuid
from typing import List

import numpy as np
from numpy.linalg import norm
from osgeo import gdal, osr
from qgis.core import QgsProject, QgsRasterLayer

from deepness.common.defines import IS_DEBUG
from deepness.common.lazy_package_loader import LazyPackageLoader
from deepness.common.misc import TMP_DIR_PATH
from deepness.common.processing_parameters.recognition_parameters import RecognitionParameters
from deepness.processing.map_processor.map_processing_result import (MapProcessingResult, MapProcessingResultCanceled,
                                                                     MapProcessingResultFailed,
                                                                     MapProcessingResultSuccess)
from deepness.processing.map_processor.map_processor_with_model import MapProcessorWithModel

cv2 = LazyPackageLoader('cv2')


class MapProcessorRecognition(MapProcessorWithModel):
    """
    MapProcessor specialized for Recognition model
    """

    def __init__(self, params: RecognitionParameters, **kwargs):
        super().__init__(params=params, model=params.model, **kwargs)
        self.recognition_parameters = params
        self.model = params.model

    def _run(self) -> MapProcessingResult:
        try:
            query_img = cv2.imread(self.recognition_parameters.query_image_path)
            assert query_img is not None, f"Error occurred while reading query image: {self.recognition_parameters.query_image_path}"
        except Exception as e:
            return MapProcessingResultFailed(f"Error occurred while reading query image: {e}")

        # some hardcoded code for recognition model
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        query_img_resized = cv2.resize(query_img, self.model.get_input_shape()[2:4][::-1])
        query_img_batched = np.array([query_img_resized])

        query_img_emb = self.model.process(query_img_batched)[0][0]

        final_shape_px = (
            self.img_size_y_pixels,
            self.img_size_x_pixels,
        )

        stride = self.stride_px
        full_result_img = np.zeros(final_shape_px, np.float32)
        mask = np.zeros_like(full_result_img, dtype=np.int16)
        highest = 0
        for tile_img_batched, tile_params_batched in self.tiles_generator_batched():
            if self.isCanceled():
                return MapProcessingResultCanceled()

            tile_result_batched = self._process_tile(tile_img_batched)[0]

            for tile_result, tile_params in zip(tile_result_batched, tile_params_batched):
                cossim = np.dot(query_img_emb, tile_result)/(norm(query_img_emb)*norm(tile_result))

                x_bin = tile_params.x_bin_number
                y_bin = tile_params.y_bin_number
                size = self.params.tile_size_px

                if cossim > highest:
                    highest = cossim
                    x_high = x_bin
                    y_high = y_bin

                full_result_img[y_bin*stride:y_bin*stride+size, x_bin*stride:x_bin*stride + size] += cossim
                mask[y_bin*stride:y_bin*stride+size, x_bin*stride:x_bin*stride + size] += 1

        full_result_img = full_result_img/mask
        self.set_results_img(full_result_img)

        gui_delegate = self._create_rlayers_from_images_for_base_extent(self.get_result_img(), x_high, y_high, size, stride)
        result_message = self._create_result_message(self.get_result_img(), x_high*self.params.tile_size_px, y_high*self.params.tile_size_px)
        return MapProcessingResultSuccess(
            message=result_message,
            gui_delegate=gui_delegate,
        )

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
            int(b.y_min * self.recognition_parameters.scale_factor): int(
                b.y_max * self.recognition_parameters.scale_factor
            ),
            int(b.x_min * self.recognition_parameters.scale_factor): int(
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
        y = y_high * stride
        x = x_high * stride

        result_img[y, x:x+size-1] = 1
        result_img[y+size-1, x:x+size-1] = 1
        result_img[y:y+size-1, x] = 1
        result_img[y:y+size-1, x+size-1] = 1

        # TODO: We are creating a new file for each layer.
        # Maybe can we pass ownership of this file to QGis?
        # Or maybe even create vlayer directly from array, without a file?

        random_id = str(uuid.uuid4()).replace("-", "")
        file_path = os.path.join(TMP_DIR_PATH, f"{random_id}.tif")
        self.save_result_img_as_tif(file_path=file_path, img=np.expand_dims(result_img, axis=2))

        rlayer = self.load_rlayer_from_file(file_path)
        OUTPUT_RLAYER_OPACITY = 0.5
        rlayer.renderer().setOpacity(OUTPUT_RLAYER_OPACITY)

        # accessing GUI from non-GUI thread is not safe, so we need to delegate it to the GUI thread
        def add_to_gui():
            group = (
                QgsProject.instance()
                .layerTreeRoot()
                .insertGroup(0, "Cosine similarity score")
            )
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

        geo_transform = [
            extent.xMinimum(),
            self.rlayer_units_per_pixel,
            0,
            extent.yMaximum(),
            0,
            -self.rlayer_units_per_pixel,
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

        # NOTE - currently we are saving result as float32, so we are losing some accuraccy.
        # result = np.clip(result, 0, 255)  # old version with uint8_t - not used anymore
        # result = result.astype(np.float32)

        return result
