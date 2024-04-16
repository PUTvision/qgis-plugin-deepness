Convert custom model to ONNX format
===================================


=======
Pytorch
=======

Steps based on `EXPORTING A MODEL FROM PYTORCH TO ONNX AND RUNNING IT USING ONNX RUNTIME <https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html>`_.

Step 0. Requirements:
  
  - Pytorch
  
  - ONNX 

Step 1. Load PyTorch model
  .. code-block::

    from torch import nn
    import torch.utils.model_zoo as model_zoo
    import torch.onnx

    model = ... # your model instation
    model.load_state_dict(torch.load(YOUR_MODEL_CHECKPOINT_PATH, map_location='cpu')['state_dict'])
    model.eval()

Step 2. Create data sample with :code:`batch_size=1` and call forward step of your model:
  .. code-block:: 

    x = torch.rand(1, INP_CHANNEL, INP_HEIGHT, INP_WIDTH) # eg. torch.rand([1, 3, 256, 256])
    _ = model(x)

Step 3a. Call export function with static batch_size=1:

  .. code-block:: 

    torch.onnx.export(model,
                    x,  # model input
                    'model.onnx',  # where to save the model
                    export_params=True,
                    opset_version=15,
                    input_names=['input'],
                    output_names=['output'],
                    do_constant_folding=False)

Step 3b. Call export function with dynamic batch_size:

  .. code-block:: 

    torch.onnx.export(model,
                    x,  # model input
                    'model.onnx',  # where to save the model
                    export_params=True,
                    opset_version=15,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'},  # variable lenght axes
                                  'output': {0: 'batch_size'}})

================
Tensorflow/Keras
================

Steps based on the `tensorflow-onnx <https://github.com/onnx/tensorflow-onnx>`_ repository. The instruction is valid for :code:`saved model` format. For other types follow :code:`tensorflow-onnx` instructions.

Requirements:
  
  - tensorflow
  
  - ONNX
  
  - tf2onnx

And simply call converter script:

  .. code-block:: 

    python -m tf2onnx.convert --saved-model YOUR_MODEL_CHECKPOINT_PATH --output model.onnx --opset 15

===============================================
Update ONNX model to support dynamic batch size
===============================================

To convert model to support dynamic batch size, you need to update :code:`model.onnx` file. You can do it manually using `this <https://github.com/onnx/onnx/issues/2182#issuecomment-881752539>`_ script. Please note that the script is not perfect and may not work for all models.
