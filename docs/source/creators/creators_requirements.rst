creators_requirements
=====

[ note copied from README, to be cleaned]

ONNX models are supported.

Model should have one input of size (BATCH_SIZE, CHANNELS, SIZE_PX, SIZE_PX).

Size of processed images in pixel is model defined, but needs to be equal in x and y axes.

Currently, BATCH_SIZE is always equal to 1.

If processed image needs to be padded (e.g. on map borders) it will be padded with 0 values.

Model output number 0 should be an image with size (BATCH_SIZE, NUM_CLASSES, SIZE_PX, SIZE_PX).

Input image data - only uint8_t value for each pixel channel.

TODO - metadata for output names and predefined processing parameters
