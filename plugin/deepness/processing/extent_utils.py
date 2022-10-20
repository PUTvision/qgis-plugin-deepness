"""
This file contains utilities related to Extent processing
"""

from qgis.core import QgsCoordinateTransform
from qgis.core import QgsRasterLayer
from qgis.core import QgsRectangle
from qgis.core import QgsVectorLayer
from qgis.gui import QgsMapCanvas

from deepness.common.errors import OperationFailedException
from deepness.common.processing_parameters.map_processing_parameters import ProcessedAreaType, \
    MapProcessingParameters
from deepness.processing.processing_utils import BoundingBox, convert_meters_to_rlayer_units


def round_extent_to_rlayer_grid(extent: QgsRectangle, rlayer: QgsRasterLayer) -> QgsRectangle:
    """
    Round to rlayer "grid" for pixels.
    Grid starts at rlayer_extent.xMinimum & yMinimum
    with resolution of rlayer_units_per_pixel

    :param extent: Extent to round, needs to be in rlayer CRS units
    :param rlayer: layer detemining the grid
    """
    grid_spacing = rlayer.rasterUnitsPerPixelX(), rlayer.rasterUnitsPerPixelY()
    grid_start = rlayer.extent().xMinimum(), rlayer.extent().yMinimum()

    x_min = grid_start[0] + int((extent.xMinimum() - grid_start[0]) / grid_spacing[0]) * grid_spacing[0]
    x_max = grid_start[0] + int((extent.xMaximum() - grid_start[0]) / grid_spacing[0]) * grid_spacing[0]
    y_min = grid_start[1] + int((extent.yMinimum() - grid_start[1]) / grid_spacing[1]) * grid_spacing[1]
    y_max = grid_start[1] + int((extent.yMaximum() - grid_start[1]) / grid_spacing[1]) * grid_spacing[1]

    new_extent = QgsRectangle(x_min, y_min, x_max, y_max)
    return new_extent


def calculate_extended_processing_extent(base_extent: QgsRectangle,
                                         params: MapProcessingParameters,
                                         rlayer: QgsVectorLayer,
                                         rlayer_units_per_pixel: float) -> QgsRectangle:
    """Calculate the "extended" processing extent, which is the full processing area, rounded to the tile size and overlap

    Parameters
    ----------
    base_extent : QgsRectangle
        Base extent of the processed ortophoto, which is not rounded to the tile size
    params : MapProcessingParameters
        Processing parameters
    rlayer : QgsVectorLayer
        mask layer
    rlayer_units_per_pixel : float
        how many rlayer CRS units are in 1 pixel

    Returns
    -------
    QgsRectangle
        The "extended" processing extent
    """

    # first try to add pixels at every border - same as half-overlap for other tiles
    additional_pixels = params.processing_overlap_px // 2
    additional_pixels_in_units = additional_pixels * rlayer_units_per_pixel

    tmp_extent = QgsRectangle(
        base_extent.xMinimum() - additional_pixels_in_units,
        base_extent.yMinimum() - additional_pixels_in_units,
        base_extent.xMaximum() + additional_pixels_in_units,
        base_extent.yMaximum() + additional_pixels_in_units,
    )

    rlayer_extent_infinite = rlayer.extent().isEmpty()  # empty extent for infinite layers
    if not rlayer_extent_infinite:
        tmp_extent = tmp_extent.intersect(rlayer.extent())

    # then add borders to have the extent be equal to  N * stride + tile_size, where N is a natural number
    tile_size_px = params.tile_size_px
    stride_px = params.processing_stride_px  # stride in pixels

    current_x_pixels = round(tmp_extent.width() / rlayer_units_per_pixel)
    if current_x_pixels <= tile_size_px:
        missing_pixels_x = tile_size_px - current_x_pixels  # just one tile
    else:
        pixels_in_last_stride_x = (current_x_pixels - tile_size_px) % stride_px
        missing_pixels_x = (stride_px - pixels_in_last_stride_x) % stride_px

    current_y_pixels = round(tmp_extent.height() / rlayer_units_per_pixel)
    if current_y_pixels <= tile_size_px:
        missing_pixels_y = tile_size_px - current_y_pixels  # just one tile
    else:
        pixels_in_last_stride_y = (current_y_pixels - tile_size_px) % stride_px
        missing_pixels_y = (stride_px - pixels_in_last_stride_y) % stride_px

    missing_pixels_x_in_units = missing_pixels_x * rlayer_units_per_pixel
    missing_pixels_y_in_units = missing_pixels_y * rlayer_units_per_pixel
    tmp_extent.setXMaximum(tmp_extent.xMaximum() + missing_pixels_x_in_units)
    tmp_extent.setYMaximum(tmp_extent.yMaximum() + missing_pixels_y_in_units)

    extended_extent = tmp_extent
    return extended_extent


def is_extent_infinite_or_too_big(rlayer: QgsRasterLayer) -> bool:
    """Check whether layer covers whole earth (infinite extent) or or is too big for processing"""
    rlayer_extent = rlayer.extent()

    # empty extent happens for infinite layers
    if rlayer_extent.isEmpty():
        return True

    rlayer_area_m2 = rlayer_extent.area() * (1 / convert_meters_to_rlayer_units(rlayer, 1)) ** 2

    if rlayer_area_m2 > (1606006962349394 // 10):  # so 1/3 of the earth (this magic value is from bing aerial map area)
        return True

    return False


def calculate_base_processing_extent_in_rlayer_crs(map_canvas: QgsMapCanvas,
                                                   rlayer: QgsRasterLayer,
                                                   vlayer_mask: QgsVectorLayer,
                                                   params: MapProcessingParameters) -> QgsRectangle:
    """ Determine the Base Extent of processing (Extent (rectangle) in which the actual required area is contained)

    Parameters
    ----------
    map_canvas : QgsMapCanvas
        currently visible map in the UI
    rlayer : QgsRasterLayer
        ortophotomap which is being processed
    vlayer_mask : QgsVectorLayer
        mask layer containing the processed area
    params : MapProcessingParameters
        Processing parameters

    Returns
    -------
    QgsRectangle
        Base Extent of processing
    """

    rlayer_extent = rlayer.extent()
    processed_area_type = params.processed_area_type
    rlayer_extent_infinite = is_extent_infinite_or_too_big(rlayer)

    if processed_area_type == ProcessedAreaType.ENTIRE_LAYER:
        expected_extent = rlayer_extent
        if rlayer_extent_infinite:
            msg = "Cannot process entire layer - layer extent is not defined or too big. " \
                  "Make sure you are not processing 'Entire layer' which covers entire earth surface!!"
            raise OperationFailedException(msg)
    elif processed_area_type == ProcessedAreaType.FROM_POLYGONS:
        expected_extent_in_vlayer_crs = vlayer_mask.extent()
        if vlayer_mask.crs() == rlayer.crs():
            expected_extent = expected_extent_in_vlayer_crs
        else:
            t = QgsCoordinateTransform()
            t.setSourceCrs(vlayer_mask.crs())
            t.setDestinationCrs(rlayer.crs())
            expected_extent = t.transform(expected_extent_in_vlayer_crs)
    elif processed_area_type == ProcessedAreaType.VISIBLE_PART:
        # transform visible extent from mapCanvas CRS to layer CRS
        active_extent_in_canvas_crs = map_canvas.extent()
        canvas_crs = map_canvas.mapSettings().destinationCrs()
        t = QgsCoordinateTransform()
        t.setSourceCrs(canvas_crs)
        t.setDestinationCrs(rlayer.crs())
        expected_extent = t.transform(active_extent_in_canvas_crs)
    else:
        raise Exception("Invalid processed are type!")

    expected_extent = round_extent_to_rlayer_grid(extent=expected_extent, rlayer=rlayer)

    if rlayer_extent_infinite:
        base_extent = expected_extent
    else:
        base_extent = expected_extent.intersect(rlayer_extent)

    return base_extent


def calculate_base_extent_bbox_in_full_image(image_size_y: int,
                                             base_extent: QgsRectangle,
                                             extended_extent: QgsRectangle,
                                             rlayer_units_per_pixel) -> BoundingBox:
    """Calculate how the base extent fits in extended_extent in terms of pixel position

    Parameters
    ----------
    image_size_y : int
        Size of the image in y axis in pixels
    base_extent : QgsRectangle
        Base Extent of processing
    extended_extent : QgsRectangle
        Extended extent of processing
    rlayer_units_per_pixel : _type_
        Number of layer units per a single image pixel

    Returns
    -------
    BoundingBox
        Bounding box describing position of base extent in the extended extent
    """
    base_extent = base_extent
    extended_extent = extended_extent

    # should round without a rest anyway, as extends are aligned to rlayer grid
    base_extent_bbox_in_full_image = BoundingBox(
        x_min=round((base_extent.xMinimum() - extended_extent.xMinimum()) / rlayer_units_per_pixel),
        y_min=image_size_y - 1 - round((base_extent.yMaximum() - extended_extent.yMinimum()) / rlayer_units_per_pixel - 1),
        x_max=round((base_extent.xMaximum() - extended_extent.xMinimum()) / rlayer_units_per_pixel) - 1,
        y_max=image_size_y - 1 - round((base_extent.yMinimum() - extended_extent.yMinimum()) / rlayer_units_per_pixel),
    )
    return base_extent_bbox_in_full_image
