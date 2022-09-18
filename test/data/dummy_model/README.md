`dummy_model.onnx` is a dummy model to be used for testing purposes (to process tiles in a well-defined way).
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
