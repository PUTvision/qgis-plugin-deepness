# qgis-deep-segmentation-framework
Plugin for QGis to perform map/image segmentation with neural network models. 

# Development
 - Install QGis `apt install qgis` (tested with QGis 3.12)
 - Create virtual environment (with global packages inherited!)
```
TODO
```
 - create a softlink to our plugin in a Qgis plugin directory:
```
ln -s $PROJECT_DIR/plugin/deep_segmentation_framework ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/deep_segmentation_framework
```
 - Activate environment and install requirements:
```
. venv/bin/activate
pip install -r requirements.txt
```
 - Run QGis in the venv:
```
qgis
```
 - Enable `Deep Segmentation Framework` the plugin in the `Plugins -> Manage and install plugins`
 - Install and enable plugin to reload plugins: `Plugin reloader`

Every time plugin code is modified, use the `Plugin reloader` to reload our plugin.