- `dummy_segmentation_model.onnx` is a dummy model to be used for testing purposes (to process tiles in a well-defined way).
Each pixel value is taken to the power of 8 (R, G and B components respectively).

So the model has 3 channels as inputs, and 3 channels as outputs.

It allows to easily detect areas with high red, green or blue color (as in the test image: `fummy_fotomap_small.tif`).

```
class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):      
        x = torch.square(x)
        x = torch.square(x)
        x = torch.square(x)
        
        return x

    def string(self):
        return f'Dymmy model'
```


- `dummy_regression_model.onnx` is a dummy model to be used for testing purposes (to process tiles in a well-defined way).
It calculates a vegetation index from RGB data, according to the equation: `VI = (2G − R − B)/(2G + R + B)`

```
class DummyRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):  
        r = x[:, 0:1, :, :]
        g = x[:, 1:2, :, :]
        b = x[:, 2:3, :, :]
        vi = (2*g - r - b) / (2*g + r + b)
        return vi

    def string(self):
        return f'Dymmy model'
```
