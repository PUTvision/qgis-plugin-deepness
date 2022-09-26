import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import cv2
from qgis._core import QgsRasterLayer
from qgis.core import Qgis
from qgis.core import QgsWkbTypes
from qgis.core import QgsRectangle

from qgis.core import QgsFeature, QgsGeometry, QgsPointXY
from qgis.core import QgsUnitTypes


from deep_segmentation_framework.common.defines import IS_DEBUG
from deep_segmentation_framework.common.processing_parameters.segmentation_parameters import SegmentationParameters
from deep_segmentation_framework.common.processing_parameters.map_processing_parameters import MapProcessingParameters

if IS_DEBUG:
    pass


def convert_meters_to_rlayer_units(rlayer, distance_m) -> float:
    """ How many map units are there in one meter """
    # TODO - potentially implement conversions from other units
    if rlayer.crs().mapUnits() != QgsUnitTypes.DistanceUnit.DistanceMeters:
        # TODO - add support for more unit types
        raise Exception("Unsupported layer units")
    assert distance_m != 0
    return distance_m


def get_numpy_data_type_for_qgis_type(data_type_qgis: Qgis.DataType):
    if data_type_qgis == Qgis.DataType.Byte:
        data_type_numpy = np.uint8
    elif data_type_qgis == Qgis.DataType.UInt16:
        data_type_numpy = np.uint16
    elif data_type_qgis == Qgis.DataType.Int16:
        data_type_numpy = np.int16
    elif data_type_qgis in [Qgis.DataType.Float32]:
        data_type_numpy = np.float32
    else:
        # TODO - maybe add support for more data types (change also the numpy type below then)
        raise Exception("Invalid input layer data type!")

    return data_type_numpy


def get_tile_image(
        rlayer: QgsRasterLayer,
        extent: QgsRectangle,
        params: MapProcessingParameters) -> np.ndarray:
    """

    :param rlayer: raster layer from which the image will be extracted
    :param extent: extent of the image to extract
    :param params:
    :return: extracted image [SIZE x SIZE x CHANNELS]. Probably RGBA channels
    """

    expected_meters_per_pixel = params.resolution_cm_per_px / 100
    expected_units_per_pixel = convert_meters_to_rlayer_units(rlayer, expected_meters_per_pixel)
    expected_units_per_pixel_2d = expected_units_per_pixel, expected_units_per_pixel
    # to get all pixels - use the 'rlayer.rasterUnitsPerPixelX()' instead of 'expected_units_per_pixel_2d'
    image_size = round((extent.width()) / expected_units_per_pixel_2d[0]), \
                 round((extent.height()) / expected_units_per_pixel_2d[1])

    # sanity check, that we gave proper extent as parameter
    assert image_size[0] == params.tile_size_px
    assert image_size[1] == params.tile_size_px

    # enable resampling
    data_provider = rlayer.dataProvider()
    if data_provider is None:
        raise Exception("Somehow invalid rlayer!")
    if hasattr(QgsRasterDataProvider, 'enableProviderResampling'):
        data_provider.enableProviderResampling(True)
    original_resampling_method = data_provider.zoomedInResamplingMethod()
    data_provider.setZoomedInResamplingMethod(data_provider.ResamplingMethod.Bilinear)
    data_provider.setZoomedOutResamplingMethod(data_provider.ResamplingMethod.Bilinear)

    def get_raster_block(band_number_):
        raster_block = rlayer.dataProvider().block(
            band_number_,
            extent,
            image_size[0], image_size[1])
        block_height, block_width = raster_block.height(), raster_block.width()
        if block_width == 0 or block_width == 0:
            raise Exception("No data on layer within the expected extent!")
        return raster_block

    input_channels_mapping = params.input_channels_mapping
    number_of_model_inputs = input_channels_mapping.get_number_of_model_inputs()
    tile_data = []

    if input_channels_mapping.are_all_inputs_standalone_bands():
        band_count = rlayer.bandCount()
        for i in range(number_of_model_inputs):
            image_channel = input_channels_mapping.get_image_channel_for_model_input(i)
            band_number = image_channel.get_band_number()
            assert band_number <= band_count  # we cannot obtain a higher band than the maximum in the image
            rb = get_raster_block(band_number)
            raw_data = rb.data()
            bytes_array = bytes(raw_data)
            data_type = rb.dataType()
            data_type_numpy = get_numpy_data_type_for_qgis_type(data_type)
            a = np.frombuffer(bytes_array, dtype=data_type_numpy)
            b = a.reshape((image_size[1], image_size[0], 1))
            tile_data.append(b)
    elif input_channels_mapping.are_all_inputs_composite_byte():
        rb = get_raster_block(1)  # the data are always in band 1
        raw_data = rb.data()
        bytes_array = bytes(raw_data)
        dt = rb.dataType()
        number_of_image_channels = input_channels_mapping.get_number_of_image_channels()
        assert number_of_image_channels == 4  # otherwise we did something wrong earlier...
        if dt != Qgis.DataType.ARGB32:
            raise Exception("Invalid input layer data type!")
        a = np.frombuffer(bytes_array, dtype=np.uint8)
        b = a.reshape((image_size[1], image_size[0], number_of_image_channels))

        for i in range(number_of_model_inputs):
            image_channel = input_channels_mapping.get_image_channel_for_model_input(i)
            byte_number = image_channel.get_byte_number()
            assert byte_number < number_of_image_channels  # we cannot get more bytes than there are
            tile_data.append(b[:, :, byte_number:byte_number+1])  # last index to keep dimension
    else:
        raise Exception("Unsupported image channels composition!")

    data_provider.setZoomedInResamplingMethod(original_resampling_method)  # restore old resampling method
    img = np.concatenate(tile_data, axis=2)
    return img


def erode_dilate_image(img, segmentation_parameters: SegmentationParameters):
    # self._show_image(img)
    if segmentation_parameters.postprocessing_dilate_erode_size:
        size = (segmentation_parameters.postprocessing_dilate_erode_size // 2) ** 2 + 1
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


def transform_points_list_xy_to_target_crs(
        points: List[Tuple],
        extent: QgsRectangle,
        rlayer_units_per_pixel: float):
    x_left = extent.xMinimum()
    y_upper = extent.yMaximum()
    points_crs = []

    for point_xy in points:
        x_crs = point_xy[0] * rlayer_units_per_pixel + x_left
        y_crs = -(point_xy[1] * rlayer_units_per_pixel - y_upper)
        points_crs.append(QgsPointXY(x_crs, y_crs))
    return points_crs


def transform_contours_yx_pixels_to_target_crs(
        contours,
        extent: QgsRectangle,
        rlayer_units_per_pixel: float):
    x_left = extent.xMinimum()
    y_upper = extent.yMaximum()

    polygons_crs = []
    for polygon_3d in contours:
        # https://stackoverflow.com/questions/33458362/opencv-findcontours-why-do-we-need-a-
        # vectorvectorpoint-to-store-the-cont
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

    def calculate_overlap_in_pixels(self, other):
        dx = min(self.x_max, other.x_max) - max(self.x_min, other.x_min)
        dy = min(self.y_max, other.y_max) - max(self.y_min, other.y_min)
        if (dx >= 0) and (dy >= 0):
            return dx * dy
        return 0

    def get_slice(self):
        roi_slice = np.s_[self.y_min:self.y_max + 1, self.x_min:self.x_max + 1]


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
