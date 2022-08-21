from dataclasses import dataclass

import numpy as np
import cv2
from qgis.core import QgsRectangle

from qgis.core import QgsFeature, QgsGeometry, QgsVectorLayer, QgsPointXY
from qgis.core import QgsUnitTypes


from deep_segmentation_framework.common.defines import PLUGIN_NAME, LOG_TAB_NAME, IS_DEBUG
from deep_segmentation_framework.common.inference_parameters import InferenceParameters, ProcessedAreaType

if IS_DEBUG:
    from matplotlib import pyplot as plt


def convert_meters_to_rlayer_units(rlayer, distance_m) -> float:
    """ How many map units are there in one meter """
    # TODO - potentially implement conversions from other units
    if rlayer.crs().mapUnits() != QgsUnitTypes.DistanceUnit.DistanceMeters:
        raise Exception("Unsupported layer units")
    return distance_m


def get_tile_image(rlayer, extent, inference_parameters: InferenceParameters) -> np.ndarray:
    """

    :param rlayer: raster layer from which the image will be extracted
    :param extent: extent of the image to extract
    :param inference_parameters:
    :return: extracted image [SIZE x SIZE x CHANNELS]. Probably RGBA channels
    """

    expected_meters_per_pixel = inference_parameters.resolution_cm_per_px / 100
    expected_units_per_pixel = convert_meters_to_rlayer_units(rlayer, expected_meters_per_pixel)
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
        block_height, block_width = rb.height(), rb.width()
        if block_width == 0 or block_width == 0:
            raise Exception("No data on layer within the expected extent!")

        raw_data = rb.data()
        bytes_array = bytes(raw_data)
        dt = rb.dataType()
        if dt == dt.__class__.Byte:
            number_of_channels = 1
        elif dt == dt.__class__.ARGB32:
            number_of_channels = 4
        else:
            raise Exception("Invalid input layer data type!")

        a = np.frombuffer(bytes_array, dtype=np.uint8)
        b = a.reshape((image_size[1], image_size[0], number_of_channels))
        band_data.append(b)

    data_provider.setZoomedInResamplingMethod(original_resampling_method)  # restore old resampling method

    # TODO - add analysis of band names, to properly set RGBA channels
    if band_count == 4:
        band_data = [band_data[0], band_data[1], band_data[2], band_data[3]]  # RGBA probably

    img = np.concatenate(band_data, axis=2)
    return img


def erode_dilate_image(img, inference_parameters):
    # self._show_image(img)
    if inference_parameters.postprocessing_dilate_erode_size:
        print('Opening...')
        size = (inference_parameters.postprocessing_dilate_erode_size // 2) ** 2 + 1
        kernel = np.ones((size, size), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        # self._show_image(img, 'opened')
    return img


def convert_cv_contours_to_features(features,
                                    cv_contours,
                                    hierarchy,
                                    current_contour_index,
                                    is_hole,
                                    current_holes):
    """
    Convert contour found with OpenCV to features accepted by QGis
    :param features:
    :param cv_contours:
    :param hierarchy:
    :param current_contour_index:
    :param is_hole:
    :param current_holes:
    :return:
    """

    if current_contour_index == -1:
        return

    while True:
        contour = cv_contours[current_contour_index]
        if len(contour) >= 3:
            first_child = hierarchy[current_contour_index][2]
            internal_holes = []
            convert_cv_contours_to_features(
                features=features,
                cv_contours=cv_contours,
                hierarchy=hierarchy,
                current_contour_index=first_child,
                is_hole=not is_hole,
                current_holes=internal_holes)

            if is_hole:
                current_holes.append(contour)
            else:
                feature = QgsFeature()
                polygon_xy_vec_vec = [
                    contour,
                    *internal_holes
                ]
                geometry = QgsGeometry.fromPolygonXY(polygon_xy_vec_vec)
                feature.setGeometry(geometry)

                # polygon = shapely.geometry.Polygon(contour, holes=internal_holes)
                features.append(feature)

        current_contour_index = hierarchy[current_contour_index][0]
        if current_contour_index == -1:
            break


def transform_contours_yx_pixels_to_target_crs(contours,
                                               extent: QgsRectangle,
                                               rlayer_units_per_pixel: float):
    x_left = extent.xMinimum()
    y_upper = extent.yMaximum()

    polygons_crs = []
    for polygon_3d in contours:
        # https://stackoverflow.com/questions/33458362/opencv-findcontours-why-do-we-need-a-vectorvectorpoint-to-store-the-cont
        polygon = polygon_3d.squeeze(axis=1)

        polygon_crs = []
        for i in range(len(polygon)):
            yx_px = polygon[i]
            x_crs = yx_px[0] * rlayer_units_per_pixel + x_left
            y_crs = -(yx_px[1] * rlayer_units_per_pixel - y_upper)
            polygon_crs.append(QgsPointXY(x_crs, y_crs))
        polygons_crs.append(polygon_crs)
    return polygons_crs


@dataclass
class BoundingBox:
    """
    Describes a bounding box rectangle.
    Similar to cv2.Rect
    """
    x_min: int
    x_max: int
    y_min: int
    y_max: int

    def get_shape(self):
        return [
            self.y_max - self.y_min + 1,
            self.x_max - self.x_min + 1
        ]
