import time

import numpy as np
import cv2

from PyQt5.QtCore import QByteArray
from PyQt5.QtGui import QPixmap
from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication, Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction
from qgis.core import QgsCoordinateTransform
from qgis.core import QgsRectangle
from qgis.core import QgsMessageLog
from qgis.core import QgsApplication
from qgis.core import QgsTask
from qgis.core import QgsProject
from qgis.gui import QgisInterface
from qgis.core import Qgis
import qgis
from qgis.utils import iface


# Initialize Qt resources from file resources.py


# Import the code for the DockWidget
import os.path


# Run in console:
# exec(open("/home/przemek/Projects/pp/qgis-deep-segmentation-framework/tmp.py").read())

rlayer = qgis.utils.iface.activeLayer()
rlayer_extent = rlayer.extent()
rlayer_units_per_pixel = rlayer.rasterUnitsPerPixelX(), rlayer.rasterUnitsPerPixelY()

active_extent_canvas_crs = iface.mapCanvas().extent()
canvas_crs = iface.mapCanvas().mapSettings().destinationCrs()

t = QgsCoordinateTransform()
t.setSourceCrs(canvas_crs)
t.setDestinationCrs(rlayer.crs())
active_extent = t.transform(active_extent_canvas_crs)

active_extent_intersect = active_extent.intersect(rlayer_extent)
# rounded to layer "grid" for pixels. Grid starts at rlayer_extent.xMinimum with resolution of rlayer_units_per_pixel
xmin = int(active_extent_intersect.xMinimum() / rlayer_units_per_pixel[0]) * rlayer_units_per_pixel[0]
xmax = int(active_extent_intersect.xMaximum() / rlayer_units_per_pixel[0]) * rlayer_units_per_pixel[0]
ymin = int(active_extent_intersect.yMinimum() / rlayer_units_per_pixel[1]) * rlayer_units_per_pixel[1]
ymax = int(active_extent_intersect.yMaximum() / rlayer_units_per_pixel[1]) * rlayer_units_per_pixel[1]
active_extent_intersect.setXMinimum(xmin)
active_extent_intersect.setXMaximum(xmax)
active_extent_intersect.setYMinimum(ymin)
active_extent_intersect.setYMaximum(ymax)

active_extent_rounded = active_extent_intersect
# to get all pixels - use the actual image resolution, or even calculate the size directly from extent and rlayer_units_per_pixel
expected_meters_per_pixel = 0.0003
expected_units_per_pixel = expected_meters_per_pixel  # for EPSG32633
image_size = int((xmax - xmin) / expected_units_per_pixel), int((ymax - ymin) / expected_units_per_pixel)


band_count = rlayer.bandCount()

band_data = []

# enable resampling
data_provider = rlayer.dataProvider()
data_provider.enableProviderResampling(True)
original_resampling_method = data_provider.zoomedInResamplingMethod()
data_provider.setZoomedInResamplingMethod(data_provider.ResamplingMethod.Bilinear)
data_provider.setZoomedOutResamplingMethod(data_provider.ResamplingMethod.Bilinear)

for band_number in range(1, band_count+1):
    raster_block = rlayer.dataProvider().block(
        band_number,
        active_extent_rounded,
        image_size[0], image_size[1])
    rb = raster_block
    rb.height(), rb.width()
    raw_data = rb.data()
    bytes_array = bytes(raw_data)
    dt = rb.dataType()
    dt
    if dt == dt.__class__.Byte:
        number_of_channels = 1
    elif dt == dt.__class__.ARGB32:
        number_of_channels = 4
    else:
        raise Exception("Invalid type!")

    a = np.frombuffer(bytes_array, dtype=np.uint8)
    b = a.reshape((image_size[1], image_size[0], number_of_channels))
    band_data.append(b)

data_provider.setZoomedInResamplingMethod(original_resampling_method)  # restore old resampling method

if band_count == 4:
    band_data = [band_data[2], band_data[1], band_data[0], band_data[3]]

img = np.concatenate(band_data, axis=2)
cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
cv2.resizeWindow('img2', 800, 800)
cv2.imshow('img2', img)
cv2.waitKey(1)
