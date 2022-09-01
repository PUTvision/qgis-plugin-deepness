import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import cv2
from qgis.core import Qgis
from qgis.core import QgsWkbTypes
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
        if dt == Qgis.DataType.Byte:
            number_of_channels = 1
        elif dt == Qgis.DataType.ARGB32:
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


def transform_polygon_with_rings_epsg_to_extended_xy_pixels(
        polygons: List[List[QgsPointXY]],
        extended_extent: QgsRectangle,
        img_size_y_pixels: int,
        rlayer_units_per_pixel: float) -> List[List[Tuple]]:
    """
    Transform coordinates polygons to pixels contours (with cv2 format), in base_extent pixels system
    :param polygons: List of tuples with two lists each (x and y points respoectively)
    :param extended_extent:
    :param img_size_y_pixels:
    :param rlayer_units_per_pixel:
    :return: 2D contours list
    """
    xy_pixel_contours = []
    for polygon in polygons:
        xy_pixel_contour = []

        x_min_epsg = extended_extent.xMinimum()
        y_min_epsg = extended_extent.yMinimum()
        y_max_pixel = img_size_y_pixels - 1  # -1 to have the max pixel, not shape
        for point_epsg in polygon:
            x_epsg, y_epsg = point_epsg
            x = round((x_epsg - x_min_epsg) / rlayer_units_per_pixel)
            y = y_max_pixel - round((y_epsg - y_min_epsg) / rlayer_units_per_pixel)
            # NOTE: here we can get pixels +-1 values, because we operate on already rounded bounding boxes
            xy_pixel_contour.append((x, y))

        # Values:
        # extended_extent.height() / rlayer_units_per_pixel, extended_extent.width() / rlayer_units_per_pixel
        # are not integers, because extents are aligned to grid, not pixels resolution

        xy_pixel_contours.append(np.asarray(xy_pixel_contour))
    return xy_pixel_contours


def create_area_mask_image(vlayer_mask,
                           extended_extent: QgsRectangle,
                           rlayer_units_per_pixel: float,
                           image_shape_yx) -> Optional[np.ndarray]:
    """
    Mask determining area to process (within extended_extent coordinates)
    None if no mask layer provided.
    """

    if vlayer_mask is None:
        return None
    img = np.zeros(shape=image_shape_yx, dtype=np.uint8)
    features = vlayer_mask.getFeatures()

    # see https://docs.qgis.org/3.22/en/docs/pyqgis_developer_cookbook/vector.html#iterating-over-vector-layer
    for feature in features:
        print("Feature ID: ", feature.id())
        geom = feature.geometry()
        geom_single_type = QgsWkbTypes.isSingleType(geom.wkbType())

        if geom.type() == QgsWkbTypes.PointGeometry:
            logging.warning("Point geometry not supported!")
        elif geom.type() == QgsWkbTypes.LineGeometry:
            logging.warning("Line geometry not supported!")
        elif geom.type() == QgsWkbTypes.PolygonGeometry:
            polygons = []
            if geom_single_type:
                polygon = geom.asPolygon()  # polygon with rings
                polygons.append(polygon)
            else:
                polygons = geom.asMultiPolygon()

            for polygon_with_rings in polygons:
                polygon_with_rings_xy = transform_polygon_with_rings_epsg_to_extended_xy_pixels(
                    polygons=polygon_with_rings,
                    extended_extent=extended_extent,
                    img_size_y_pixels=image_shape_yx[0],
                    rlayer_units_per_pixel=rlayer_units_per_pixel)
                # first polygon is actual polygon
                cv2.fillPoly(img, pts=polygon_with_rings_xy[:1], color=255)
                if len(polygon_with_rings_xy) > 1:  # further polygons are rings
                    cv2.fillPoly(img, pts=polygon_with_rings_xy[1:], color=0)
        else:
            print("Unknown or invalid geometry")

    return img
