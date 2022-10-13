<p align="center">
  <img width="250" height="250" src="plugin/deepness/images/icon.png" alt="dsf_logo">

  <h2 align="center">Deepness: <b>Deep</b> <b>N</b>eural r<b>E</b>mote <b>S</b>en<b>S</b>ing QGIS Plugin</h2>
</p>

#

![main](https://github.com/PUTvision/qgis-plugin-deepness/actions/workflows/python-app.yml/badge.svg)
[![GitHub contributors](https://img.shields.io/github/contributors/PUTvision/qgis-plugin-deepness)](https://github.com/PUTvision/qgis-plugin-deepness/graphs/contributors)
[![GitHub stars](https://img.shields.io/github/stars/PUTvision/qgis-plugin-deepness)](https://github.com/PUTvision/qgis-plugin-deepness/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/PUTvision/qgis-plugin-deepness)](https://github.com/PUTvision/qgis-plugin-deepness/network/members)

Plugin for QGis to perform map/image segmentation with neural network models. 

# Development
 - Install QGis `apt install qgis` (tested with QGis 3.12)
 - Create virtual environment (with global packages inherited!)
```
python3 -m venv venv --system-site-packages
```
 - create a softlink to our plugin in a Qgis plugin directory:
```
ln -s $PROJECT_DIR/plugin/deepness ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/deepness
```
 - Activate environment and install requirements:
```
. venv/bin/activate
pip install -r requirements.txt
```
 - Run QGis in the venv:
```
export IS_DEBUG=true  # to enable some debugging options
qgis
```
 - Enable `Deepness` the plugin in the `Plugins -> Manage and install plugins`
 - Install and enable plugin to reload plugins: `Plugin reloader` and to print stack errors: `first aid` plugin

Every time plugin code is modified, use the `Plugin reloader` to reload our plugin.

# Unit tests
See [test/README.md](test/README.md)

# Documentation
See [docs/README.md](docs/README.md)

# Development notes
 - plugin skeleton was initially generated with `Plugin Builder`, but then refactored and cleaned up a little bit
 - Before release: change version number in `metadata.txt` and in docs (?)
 - to recreate resource file (`resource.qrsc`) run:
    ```
    cd plugin/deepness
    pyrcc5 -o resources.py resources.qrc
    ```
   Though I'm not sure if this file is even needed anymore
 - 
