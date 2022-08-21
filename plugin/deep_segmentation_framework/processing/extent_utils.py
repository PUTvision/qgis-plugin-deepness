from qgis.core import QgsRasterLayer
from qgis.core import QgsRectangle


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
