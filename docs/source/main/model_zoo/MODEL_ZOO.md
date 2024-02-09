# Deepness Model ZOO

The [Model ZOO](https://chmura.put.poznan.pl/s/2pJk4izRurzQwu3) is a collection of pre-trained, deep learning models in the ONNX format. It allows for an easy-to-use start with the plugin.

 > NOTE: the provided models are not universal tools and will perform well only on similar data as in the training datasets. If you notice the model is not perfroming well on your data, consider re-training (or fine-tuning) it on your data.

> If you do not have machine learning expertise, feel free to contact the plugin authors for help or advice.

## Segmentation models

| Model                                                                            | Input size | CM/PX | Description                                                                                                                                                                                                                                                                                                                                                                         | Example image |
|----------------------------------------------------------------------------------|------------|-------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| [Corn Field Damage Segmentation](https://chmura.put.poznan.pl/s/abWFTVYSDIcncWs) | 512        | 3     | [PUT Vision](https://putvision.github.io/) model for Corn Field Damage Segmentation created on own dataset labeled by experts. We used the classical UNet++ model. It generates 3 outputs: healthy crop, damaged crop, and out-of-field area.                                                                                                                                       | [Image](https://chmura.put.poznan.pl/s/i5WVmcfqPNdBTAQ) |
| [Land Cover Segmentation](https://chmura.put.poznan.pl/s/PnAFJw27uneROkV)        | 512        | 40    | The model is trained on the [LandCover.ai dataset](https://landcover.ai.linuxpolska.com/). It provides satellite images with 25 cm/px and 50 cm/px resolution. Annotation masks for the following classes are provided for the images: building (1), woodland (2), water(3), road(4). We use `DeepLabV3+` model with `tu-semnasnet_100` backend and `FocalDice` as a loss function. NOTE: the dataset covers only the area of Poland, therefore the performance may be inferior in other parts of the world. | [Image](https://chmura.put.poznan.pl/s/Xa29vnieNQTvSt5) |
| [Buildings Segmentation](https://chmura.put.poznan.pl/s/MwhgQNhyQF3fuBs)         | 256        | 40    | Trained on the [RampDataset dataset](https://cmr.earthdata.nasa.gov/search/concepts/C2781412367-MLHUB.html). Annotation masks for buildings and background. Xunet network. Val F1-score 81.0 | [Image](https://chmura.put.poznan.pl/s/XCjuDKDS3FFovDl) |
| [Land Cover Segmentation Sentinel-2](https://chmura.put.poznan.pl/s/UbljXBr1XSc9hCL) | 64         | 1000  | Trained on the [Eurosat dataset](https://www.tensorflow.org/datasets/catalog/eurosat). Uses 13 spectral bands from Sentinel-2, with 10 classes. Model ConvNeXt.  | [Image](https://chmura.put.poznan.pl/s/pGR5VX6AV3hYKVl) |
| [Agriculture segmentation RGB+NIR](https://chmura.put.poznan.pl/s/wf5Ml1ZDyiVdNiy) | 256        | 30    | Trained on the [Agriculture Vision 2021 dataset](https://www.agriculture-vision.com/agriculture-vision-2021/dataset-2021). 4 channels input (RGB + NIR). 9 output classes within agricultural field (weed_cluster, waterway, ...). Uses X-UNet. | [Image](https://chmura.put.poznan.pl/s/35A5ISUxLxcK7kL) |
| [Fire risk assesment](https://chmura.put.poznan.pl/s/NxKLdfdr9s9jsVA)            | 384        | 100   | Trained on the FireRisk dataset (RGB data). Classifies risk of fires (ver_high, high, low, ...). Uses ConvNeXt XXL. Val F1-score 	65.5. | [Image](https://chmura.put.poznan.pl/s/Ijn3VgG76NvYtDY) |
| [Roads Segmentation](https://chmura.put.poznan.pl/s/y6S3CmodPy1fYYz)             | 512        | 21    | The model segments the Google Earth satellite images into 'road' and 'not-road' classes. Model works best on wide car roads, crossroads and roundabouts.                                                                                                                                                                                                                            | [Image](https://chmura.put.poznan.pl/s/rln6mpbjpsXWpKg) |

## Regression models

| Model   | Input size | CM/PX | Description | Example image |
|---------|---|---|---|---|
|         |  |  |  |  |

## Recognition models

| Model   | Input size | CM/PX | Description | Example image |
|---------|---|---|---|---|
|  [NAIP Place recognition](https://chmura.put.poznan.pl/s/k7EvbNGc2udHvck) | 224 | 100 | ConvNeXt nano trained using SimSiam onn [NAIP imagery](https://earth.esa.int/eogateway/catalog/pleiades-esa-archive). Rank1-accuracy 75.0. | [Image](https://chmura.put.poznan.pl/s/UzAvz8w5ceCui9y) |
|         |  |  |  |  |

## Object detection models

| Model                                                                          | Input size | CM/PX | Description                                                                                                                                                                                   | Example image                                           |
|--------------------------------------------------------------------------------|------------|-------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
| [Airbus Planes Detection](https://chmura.put.poznan.pl/s/bBIJ5FDPgyQvJ49)      | 256        | 70    | YOLOv7 tiny model for object detection on satellite images. Based on the [Airbus Aircraft Detection dataset](https://www.kaggle.com/datasets/airbusgeo/airbus-aircrafts-sample-dataset).      | [Image](https://chmura.put.poznan.pl/s/VfLmcWhvWf0UJfI) |
| [Airbus Oil Storage Detection](https://chmura.put.poznan.pl/s/gMundpKsYUC7sNb) | 512        | 150   | YOLOv5-m model for object detection on satellite images. Based on the [Airbus Oil Storage Detection dataset](https://www.kaggle.com/datasets/airbusgeo/airbus-oil-storage-detection-dataset). | [Image](https://chmura.put.poznan.pl/s/T3pwaKlbFDBB2C3) |
| [Aerial Cars Detection](https://chmura.put.poznan.pl/s/vgOeUN4H4tGsrGm)        | 640        | 10    | YOLOv7-m model for cars detection on aerial images. Based on the [ITCVD](https://arxiv.org/pdf/1801.07339.pdf).                                                                               | [Image](https://chmura.put.poznan.pl/s/cPzw1mkXlprSUIJ) |
| [UAVVaste Instance Segmentation](https://chmura.put.poznan.pl/s/v99rDlSPbyNpOCH)        | 640        | 0.5    | YOLOv8-L Instance Segmentation model for litter detection on high-quality UAV images. Based on the [UAVVaste dataset](https://github.com/PUTvision/UAVVaste).                                                                               | [Image](https://chmura.put.poznan.pl/s/KFQTlS2qtVnaG0q) |

## Super Resolution Models
| Model                                                                          | Input size | CM/PX | Scale Factor |Description                                                                                                                                                                                   | Example image                                           |
|--------------------------------------------------------------------------------|------------|-------|------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------|
|[Residual Dense Network (RDN X2)](https://chmura.put.poznan.pl/s/cLBZpjYn3ubuoii)      |64      |Trained on 10 cm/px images set it same as input data   | X2 |   Model originally trained by H Zhang et. al. in "[A Comparative Study on CNN-Based Single-Image Super-Resolution Techniques for Satellite Images](https://github.com/farahmand-m/satellite-image-super-resolution)" converted to onnx format   | [Image](https://chmura.put.poznan.pl/s/Ruz24ZpMNg97joV) from Massachusetts Roads Dataset [Dataset in kaggle](https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset) |
|[Residual Dense Network (RDN X4)](https://chmura.put.poznan.pl/s/AaKySmOoOhxW6qZ)      |64      |Trained on 10 cm/px images set it same as input data   | X4 |   Model originally trained by H Zhang et. al. in "[A Comparative Study on CNN-Based Single-Image Super-Resolution Techniques for Satellite Images](https://github.com/farahmand-m/satellite-image-super-resolution)" converted to onnx format   | [Image](https://chmura.put.poznan.pl/s/Ruz24ZpMNg97joV) from Massachusetts Roads Dataset [Dataset in kaggle](https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset) |


## Contributing

PRs with models are welcome!

* Please follow the [general model information](https://qgis-plugin-deepness.readthedocs.io/en/latest/creators/creators_description_classes.html).

* Use `MODEL_ZOO` tag in your PRs to make it easier to find them.

* If you need, you can check [how to export the model to ONNX](https://qgis-plugin-deepness.readthedocs.io/en/latest/creators/creators_example_onnx_model.html).

* And do not forget to [add metadata to the ONNX model](https://qgis-plugin-deepness.readthedocs.io/en/latest/creators/creators_add_metadata_to_model.html).

* You can host your model yourself or ask us to do it.
