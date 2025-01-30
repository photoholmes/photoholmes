# Datasets Module

## Table of Contents

- [Datasets Module](#datasets-module)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Available Datasets](#available-datasets)
  - [Structure](#structure)
    - [BaseDataset](#basedataset)
    - [Custom Datasets](#custom-datasets)
    - [Dataset Factory](#dataset-factory)
    - [Dataset Registry](#dataset-registry)
  - [Examples of Use](#examples-of-use)
    - [Importing the dataset directly:](#importing-the-dataset-directly)
    - [Using the DatasetFactory:](#using-the-datasetfactory)
  - [Datasets Description](#datasets-description)
    - [Columbia](#columbia)
    - [Coverage](#coverage)
    - [DSO-1](#dso-1)
    - [Korus](#korus)
    - [Casia 1.0](#casia-10)
    - [AutoSplice](#autosplice)
    - [Trace](#trace)
  - [Contribute: Adding a new dataset](#contribute-adding-a-new-dataset)

## Overview

This module provides a collection of datasets that can be used to test the performance of the methods in the `methods` module.
The different datasets are selected to cover a wide range of image manipulation techniques and to provide a good benchmark for the methods.

The datasets cover a wide range of forgery types as well as image formats, which we deemed important to benchmark the diverse array of included methods. For some datasets, there is also a version traversed through social media which is included in the library. In addition, we included a WebP version of the Korus dataset since, to our knowledge, no forgery detection dataset features this increasingly popular format.

## Available Datasets

The following datasets are available in the PhotoHolmes library:

- Columbia: A dataset of spliced images.
- Coverage: A dataset of copy-move manipulated images.
- DSO-1: A dataset of spliced images.
- Korus: A dataset of spliced, copy-move, and object removal manipulated images.
- Casia 1.0: A dataset of spliced and copy-move manipulated images.
- AutoSplice: A dataset of generative inpainting manipulated images.
- Trace: A dataset of images with alterations to the acquisition pipeline.

The following table provides an overview of the datasets and their characteristics:

| Dataset | Types of Forgery | Nb. of Images (🔵 forged + 🟠 pristine) | Format | Social Media Version | WebP Version |
|---------|------------------|----------------------------------------|--------|----------------------|--------------|
| [Columbia](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/dlform.html) | Splicing | 363 (🔵180 + 🟠183) | TIF | ✅ | ❌ |
| [Coverage](https://github.com/wenbihan/coverage) | Copy-move | 200 (🔵100 + 🟠100) | TIF | ❌ | ❌ |
| [DSO-1](https://recodbr.wordpress.com/code-n-data/#dso1_dsi1) | Splicing | 200 (🔵100 + 🟠100) | PNG | ✅ | ❌ |
| [Korus](https://pkorus.pl/downloads/dataset-realistic-tampering) | Splicing, copy-move, object removal | 440 (🔵220 + 🟠220) | TIF | ❌ | ✅ |
| [Casia 1.0](https://github.com/namtpham/casia1groundtruth/tree/master) | Splicing, copy-move | 1023 (🔵923 + 🟠100) | JPEG | ✅ | ❌ |
| [AutoSplice](https://github.com/shanface33/AutoSplice_Dataset?tab=readme-ov-file) | Generative inpainting | 5894 (🔵3621 + 🟠2273) | JPEG | ❌ | ❌ |
| [Trace](https://github.com/qbammey/trace) | Alterations to acquisition pipeline | 24000 (🔵24000 + 🟠0) | PNG | ❌ | ❌ |

The datasets are described in more detail in the [Datasets Description](#datasets-description) section.

## Structure

The datasets are structured in the following way:

### BaseDataset

The `BaseDataset` class takes care of loading the images data and the masks from the dataset. It does so by delegating to the child classes the implementation of _\_get\_paths_ that obtains the list of the image and masks paths according to a specific dataset.

Functionalities:

- It also takes care of preprocessing the images data if a preprocessing pipeline is provided.
- You can load the following image data from the datasets:
  - Image: The original image.
  - DCT Coefficients: The DCT coefficients of the image.
  - Q Tables: The Q tables of the image.
- You can choose to load tampered only images or tampered and pristine images.
- The name of the image is also retrieved from the path of the image. This is useful for the evaluation of the methods and saving the results.
- Mask binarization (often overriden).

### Custom Datasets

The datasets are structured in the following way:

- dataset.py file: Contains the class that inherits from the BaseDataset class. This class has at least two methods and declares two attributes:
  - `_get_paths`: that returns the paths to the images and masks in the dataset.
  - `_get_mask_path`: that returns the path to the mask given an image path.
  - `IMAGE_EXTENSION`: the extension of the images in the dataset.
  - `MASK_EXTENSION`: the extension of the masks in the dataset.

The two methods are used to get the paths to the images and the masks in the dataset. As different datasets have different directory structures, these methods are implemented in each dataset class. Pristine images must yield 'None' as a mask path, which is interpreted by the Benchmark as an all-zero mask with the shape of the image. In the case of the two attributes, they are used to show warning messages when jpeg data is requested from a dataset that does not have jpeg images or masks.

It is also common to override the `binarize_mask`, as different datasets have different ways to represent masks as RGB images or represent tampering in multiple levels. The PhotoHolmes library expects masks as boolean tensors of one channel, regarding any degree of tampering as _True_.

### Dataset Factory

The `DatasetFactory` class provides a simple way of loading the datasets. It has a method called `load` that takes the name of the dataset, the path to the dataset, an optional preprocessing pipeline, an optional flag to load only tampered images, and a parameter indicating which image data to load. It returns the dataset instanced with the inputed preprocessing pipeline and parameters.

### Dataset Registry

The `DatasetRegistry` class is a registry of the available datasets. It contains a dictionary with the names of the datasets as keys and the corresponding dataset classes as values. This class is used by the `DatasetFactory` to load the datasets.

You can get all the available datasets by calling the `get_all_datasets` method from the `DatasetRegistry` class.

## Examples of Use

Here are some examples of how to use the datasets:

### Importing the dataset directly:

You can easily use the chosen dataset by importing the dataset directly from the PhotoHolmes library.

```python
from photoholmes.datasets.columbia import ColumbiaDataset
from photoholmes.utils.image import plot

# Load the dataset
dataset_path = "dataset_path"
dataset = ColumbiaDataset(
    dataset_path=dataset_path,
    preprocessing_pipeline=None,
    tampered_only=True,
    load=["image"]
)

# Get the first image
data, mask, image_name = dataset[0]
image = data["image"]
plot(image)
```

For loading the pristine images as well you can do the following:

```python
from photoholmes.datasets.columbia import ColumbiaDataset
from photoholmes.utils.image import plot

# Load the dataset
dataset_path = "dataset_path"
dataset = ColumbiaDataset(
    dataset_path=dataset_path,
    preprocessing_pipeline=None,
    tampered_only=False,
    load=["image"]
)

# Get the first image
data, mask, image_name = dataset[0]
image = data["image"]
plot(image)
```

For iterating over the dataset you can do the following:

```python
from photoholmes.datasets.columbia import ColumbiaDataset
from photoholmes.utils.image import plot_multiple

# Load the dataset
dataset_path = "dataset_path"
dataset = ColumbiaDataset(
    dataset_path=dataset_path,
    preprocessing_pipeline=None,
    tampered_only=True,
    load=["image"]
)

# Iterate over the dataset and plot the images
images = []
for data, mask, image_name in dataset:
    images.append(data["image"])
    if len(images) == 4:
        break
plot_multiple(images)
```

For also loading the DCT coefficients and Q tables you can do the following:

```python
from photoholmes.datasets.columbia import ColumbiaDataset
from photoholmes.utils.image import plot
# Load the dataset
dataset_path = "dataset_path"
dataset = ColumbiaDataset(
    dataset_path=dataset_path,
    preprocessing_pipeline=None,
    tampered_only=True,
    load=["image", "dct_coefficients", "qtables"]
)

# Get the first image, the DCT coefficients and the Q tables
data, mask, image_name = dataset[0]
image = data["image"]
dct_coefficients = data["dct_coefficients"]
qtables = data["qtables"]
plot(image)
print("dct_coefficients:", dct_coefficients)
print("qtables:", qtables)
```

### Using the DatasetFactory:

You can also use the DatasetFactory to import the dataset. Here is an example of how to use the DatasetFactory with the Columbia dataset:

```python
# Import the DatasetFactory
from photoholmes.datasets.factory import DatasetFactory
from photoholmes.utils.image import plot

# Use the DatasetFactory to import the dataset
dataset_name = "columbia"
dataset_path = "dataset_path"
preprocessing_pipeline = None
tampered_only = True
load = ["image"]

dataset = DatasetFactory.load(
    dataset_name=dataset_name,
    dataset_path=dataset_path,
    preprocessing_pipeline=preprocessing_pipeline,
    tampered_only=tampered_only,
    load=load,
)

# Get the first image
data, mask, image_name = dataset[0]
image = data["image"]
plot(image)
```

Using the `DatasetRegistry` you can also get the dataset by name:

```python
from photoholmes.datasets.factory import DatasetFactory
from photoholmes.datasets.registry import DatasetRegistry
from photoholmes.utils.image import plot

# Get the dataset by name
dataset_path = "dataset_path"
preprocessing_pipeline = None
tampered_only = True
load = ["image"]

dataset = DatasetFactory.load(
    dataset_name=DatasetRegistry.COLUMBIA,
    dataset_path=dataset_path,
    preprocessing_pipeline=preprocessing_pipeline,
    tampered_only=tampered_only,
    load=load,
)

# Get the first image
data, mask, image_name = dataset[0]
image = data["image"]
plot(image)
```_get_paths

## Using a PreProcessingPipeline:

You can also use a `PreProcessingPipeline` from  [`PhotoHolmes Preprocessing Module`](../preprocessing/README.md) to preprocess the images before using them. Here is an example of how to use a preprocessing pipeline with the Columbia dataset:

```python
from photoholmes.datasets.columbia import ColumbiaDataset
from photoholmes.methods.zero import zero_preprocessing
from photoholmes.utils.image import plot

# Load the dataset
dataset_path = "dataset_path"
dataset = ColumbiaDataset(
    dataset_path=dataset_path,
    preprocessing_pipeline=zero_preprocessing,
    tampered_only=True,
    load=["image"],
)

# Get the first image
data, mask, image_name = dataset[0]
image = data["image"]
plot(image)
```

## Datasets Description

Here is a brief description of the datasets and their characteristics:

### Columbia

This dataset contains spliced images, which are not realistic at all and could be easily detected by semantic evaluation. This means that just by looking at the image and considering the context, a person can identify the suspicious area. One could argue that detecting forgeries of this type does not add value to a method, as they can be easily identified by the human eye. However, the importance of this dataset lies not only in its popularity but also in the fact that it has its version through different social networks. With the correct metrics, it allows for the quantification of how well or poorly a method can generalize in the wild forgeries, especially in the context of the different processing an image undergoes when uploaded to any social network. 

### Coverage

It is the most popular dataset for evaluating copy-move forgeries. The images in this dataset are uncompressed, and the pristine images consistently feature a repetition of a certain object. For the forged images, one of these objects is cut and pasted elsewhere, with the pasted object sometimes easily located and other times not. This dataset helps determine whether a method merely searches for similar parts within the image to detect a copy-move forgery or if it looks for inconsistencies in traces, such as the demosaicing grid.

### DSO-1

DSO-1 is a dataset that contains spliced images in which the subject used for the splicing are humans. At first glance, the splices are hard to catch, however most of the times, doing a semantic evaluation regarding the light shows which subject is spliced. This dataset is of PNG images and it has its version through different social networks.

### Korus

The Korus dataset is also named realistic tampering. As the title suggests, this dataset contains forgeries that are almost impossible to detect through semantic evaluation. It has uncompressed images containing splicing copy move and object removal.

### Casia 1.0

This dataset contains both splicing and copy move forgeries which are not so easy to identify to the naked eye and are JPEG compressed. It also has its version through different social networks which allows the same analysis as Columbia on top of being spliced and copy move forgeries JPEG compressed.

### AutoSplice

This novel dataset is unique as it incorporates generative inpainting. Jia et al. introduce the utilization of to generate forged images guided by a text prompt. These images are JPEG compressed, and the dataset includes variations with three JPEG quality factors: 100, 90, and 75. This diversity facilitates the quantification of how well methods can handle varying degrees of JPEG compression.

### Trace

In Trace, the forged and pristine regions differ only in the traces left behind by the imaging pipeline. The concept involves selecting a raw image and processing it using two distinct imaging pipelines. The results are then merged, forming a single image with two areas, each corresponding to one of the two pipelines. The merging of these images is accomplished using a mask.

## Contribute: Adding a new dataset

1. Create a new file for the dataset in the datasets folder.
2. Add the dataset to the registry and to the factory.
3. Fill out the README and don't forget to include the characteristics of the dataset.
4. Make a pull request to the repository with the new dataset following the instructions of the [CONTRIBUTING.md](../../../CONTRIBUTING.md) file.
