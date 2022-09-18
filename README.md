# ![dsf_logo](./plugin/deep_segmentation_framework/icon.png) QGIS: Deep Segmentation Framework
![main](https://github.com/PUTvision/qgis-deep-segmentation-framework/actions/workflows/python-app.yml/badge.svg)
[![GitHub contributors](https://img.shields.io/github/contributors/PUTvision/qgis-deep-segmentation-framework)](https://github.com/PUTvision/qgis-deep-segmentation-framework/graphs/contributors)
[![GitHub stars](https://img.shields.io/github/stars/PUTvision/qgis-deep-segmentation-framework)](https://github.com/PUTvision/qgis-deep-segmentation-framework/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/PUTvision/qgis-deep-segmentation-framework)](https://github.com/PUTvision/qgis-deep-segmentation-framework/network/members)

Plugin for QGis to perform map/image segmentation with neural network models. 

# Development
 - Install QGis `apt install qgis` (tested with QGis 3.12)
 - Create virtual environment (with global packages inherited!)
```
python3 -m venv venv --system-site-packages
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
 - Install and enable plugin to reload plugins: `Plugin reloader` and to print stack errors: `first aid` plugin

Every time plugin code is modified, use the `Plugin reloader` to reload our plugin.

# Unit tests
See [test/README.md](test/README.md)

# Model requirements
ONNX models are supported.
Model should have one input of size (BATCH_SIZE, CHANNELS, SIZE_PX, SIZE_PX).
Size of processed images in pixel is model defined, but needs to be equal in x and y axes.
Currently, BATCH_SIZE is always equal to 1.
If processed image needs to be padded (e.g. on map borders) it will be padded with 0 values.
Model output number 0 should be an image with size (BATCH_SIZE, NUM_CLASSES, SIZE_PX, SIZE_PX).
Input image data - only uint8_t value for each pixel channel.
