import time

import numpy as np
import cv2

from PyQt5.QtCore import QByteArray
from PyQt5.QtGui import QPixmap
from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication, Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction
from qgis._core import QgsVectorLayer, QgsVectorLayerUtils, QgsGeometry, QgsPoint, QgsFeature, QgsPointXY
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

rlayer = iface.activeLayer()


for layer_id, layer in QgsProject.instance().mapLayers().items():
    if layer.name() == 'model_output':
        QgsProject.instance().removeMapLayer(layer_id)
        break


vlayer = QgsVectorLayer("multipolygon", "model_output", "memory")
vlayer.setCrs(rlayer.crs())
prov = vlayer.dataProvider()


# feat = QgsFeature()
# geometry = QgsGeometry.fromPolygonXY([[
#     QgsPointXY(638876, 5802443),
#     QgsPointXY(638896, 5802443),
#     QgsPointXY(638896, 5802463),
# ]])
# feat.setGeometry(geometry)
# prov.addFeatures([feat])


feat = QgsFeature()
geometry = QgsGeometry.fromPolygonXY([
    [
        QgsPointXY(638856, 5802423),
        QgsPointXY(638886, 5802453),
        QgsPointXY(638899, 5802403),
    ],
    [
        QgsPointXY(638880, 5802438),
        QgsPointXY(638870, 5802428),
        QgsPointXY(638890, 5802430),
    ],
    [
        QgsPointXY(638875.89, 5802423.04),
        QgsPointXY(638890.14, 5802421.42),
        QgsPointXY(638888.92, 5802413.58),
    ],
])
# geometry.addRing([
#     QgsPointXY(638880, 5802438),
#     QgsPointXY(638870, 5802428),
#     QgsPointXY(638890, 5802430),
# ])
feat.setGeometry(geometry)
prov.addFeatures([feat])


vlayer.updateExtents()
QgsProject.instance().addMapLayer(vlayer)



"""
f = next(iface.activeLayer().getFeatures())
geom = f.geometry()
x = geom.asMultiPolygon()

geom = next(iface.activeLayer().getFeatures()).geometry()
poly = geom.asPolygon()
"""
