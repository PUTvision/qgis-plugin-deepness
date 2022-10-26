"""
This script allows you to add metadata to a model file in ONNX.
These parameters are saved in the model metadata (inside the model file) and can be read in the plugin UI.
"""

import json
import onnx


model = onnx.load('/path/to/model')

class_names = {
    0: 'not_road',
    1: 'road',
}

m = model.metadata_props.add()
m.key = 'model_type'
m.value = json.dumps('Segmentor')


m = model.metadata_props.add()
m.key = 'class_names'
m.value = json.dumps(class_names)

m = model.metadata_props.add()
m.key = 'resolution'
m.value = json.dumps(21)

m = model.metadata_props.add()
m.key = 'tiles_size'
m.value = json.dumps(512)

m = model.metadata_props.add()
m.key = 'tiles_overlap'
m.value = json.dumps(15)

m = model.metadata_props.add()
m.key = 'seg_thresh'
m.value = json.dumps(0.5)

m = model.metadata_props.add()
m.key = 'seg_small_segment'
m.value = json.dumps(11)


onnx.save(model, 'path/where/the/model/with/metadata/will/be/saved')
