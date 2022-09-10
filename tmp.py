from qgis._core import QgsVectorLayer, QgsGeometry, QgsFeature, QgsPointXY
from qgis.core import QgsProject
from qgis.utils import iface

# Initialize Qt resources from file resources.py


# Import the code for the DockWidget

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
