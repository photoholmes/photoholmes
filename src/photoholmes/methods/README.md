# Methods Module

## Table of Contents

- [Methods Module](#methods-module)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Available Methods](#available-methods)
  - [Structure](#structure)
  - [Method Factory](#method-factory)
  - [Examples of Use](#examples-of-use)
    - [Importing the method directly:](#importing-the-method-directly)
    - [Using the MethodFactory:](#using-the-methodfactory)
  - [Create a method in the PhotoHolmes standard](#create-a-method-in-the-photoholmes-standard)
    - [Create the method folder and source files](#create-the-method-folder-and-source-files)
    - [Fill all of the corresponding files described in the Structure section](#fill-all-of-the-corresponding-files-described-in-the-structure-section)
      - [The method file](#the-method-file)
      - [The preprocessing file](#the-preprocessing-file)
      - [A config file](#a-config-file)
  - [Contribute: Adding a new method to the library](#contribute-adding-a-new-method-to-the-library)
    - [Add to the MethodRegistry](#add-to-the-methodregistry)
    - [Add to the MethodFactory](#add-to-the-methodfactory)
    - [Fill out a README file](#fill-out-a-readme-file)
    - [Pull request to the repository](#pull-request-to-the-repository)

## Overview

This module provides a collection of methods that can be used by themselves to make predictions on suspicious images and that can be used with the benchmark module to check their performance with the available datasets.

## Available Methods

In the first version of the PhotoHolmes library 10 methods are available:

- __Adaptive CFA Net__: An Adaptive Neural Network for Unsupervised Mosaic Consistency Analysis in Image Forensics.
- __CAT-Net__: Compression Artifact Tracing Network for Detection and Localization of Image Splicing.
- __DQ__: Fast, Automatic, and Fine-Grained Tampered JPEG Image Detection via DCT Coefficient Analysis.
- __EXIF as Language__: Learning Cross-Modal Associations Between Images and Camera Metadata.
- __FOCAL__: Rethinking Image Forgery Detection via Contrastive Learning and Unsupervised Clustering.
- __Noisesniffer__: Image forgery detection based on noise inspection: analysis and refinement of the Noisesniffer method.
- __PSCC-Net__: Progressive Spatio-Channel Correlation Network for Image Manipulation Detection and Localization.
- __Splicebuster__: a new blind image splicing detector.
- __TruFor__: Leveraging all-round clues for trustworthy image forgery detection and localization.
- __ZERO__: A Local JPEG Grid Origin Detector Based on the Number of DCT Zeros and its Applications in Image Forensics.

For more information regarding the nature of each method please refer to their corresponding README.

## Structure

Methods in the PhotoHolmes library consist of at least these parts:

- `method.py`: Contains the method class that inherits from the `BaseMethod` class by default. In the case of methods which are end-to-end torch modules, these inherit from the `BaseTorchMethod` class. The child class of the method must define at least the following three methods:
  - `__init__`: that initializes the class. It is important to begin by calling the `__init__` of the parent class.
  - `predict`: that returns the original output of the method for a given image.
  - `benchmark`: that for a given an image, returns a standardized `BenchmarkOutput` (which is convenient for the `Benchmark` class). This output consists of a dictionary accepting only the keys of `heatmap`, `mask`, `detection` and `extra_outputs`, which are expected to contain the corresponding values.
- `preprocessing.py`: Contains the preprocessing pipeline needed for each method.
- `config.yaml`: YAML file that contains the example config for each method with default parameters.

There are some additional files that may be included in a standard form, such as:

- `config.py`: to outline the architecture configuration in deep learning modules. These are commonly implemented in the form of dataclasses.
- `postprocessing.py`: for functions that are used in the postprocessing of the mask in order to yield the expected output in the `predict` method. Note that these are of internal use of the method and should be invoked within the `predict` method (unlike the standard `preprocessing` module previously mentioned).
- `utils.py`: containing useful functions used in the method's prediction.

## Method Factory

The `MethodFactory` class provides a way of loading the method and its corresponding preprocessing.

It returns a Tuple containing the method object and the preprocessing pipeline.

## Examples of Use

Here are some examples of how to use the methods in this module:

### Importing the method directly:

You can easily use the chosen method by importing the method directly from the PhotoHolmes library.
If the method is not deep learning based it will look like this:

```python
from photoholmes.methods.chosen_method import Method, method_preprocessing

# Read an image
from photoholmes.utils.image import read_image
path_to_image = "path_to_image"
image = read_image(path_to_image)

# Assign the image to a dictionary and preprocess the image
image_data = {"image": image}
input_data = method_preprocessing(**image_data)

# Declare the method
method = Method()

# Use predict to get the final result
output = method.predict(**input_data)
```

If the method is deep learning based it will look like this:

```python
from photoholmes.methods.chosen_method import Method, method_preprocessing

# Read an image
from photoholmes.utils.image import read_image
path_to_image = "path_to_image"
image = read_image(path_to_image)

# Assign the image to a dictionary and preprocess the image
image_data = {"image": image}
input = method_preprocessing(**image_data)

# Declare the method and use the .to_device if you want to run it on cuda or mps instead of cpu
path_to_weights = "path_to_weights"
method = Method(
    weights= path_to_weights
)
device = "cpu"
method.to_device(device)

# Use predict to get the final result
output = method.predict(**input)
```

For the deep learning based methods you will need the corresponding weights. For information on where to find them please refer to the method README.

Bear in mind that the required input for a method may be more than just the input image, as it is specific of each method. Once again, each method's README contains a description of how its input should look like.

### Using the MethodFactory:

```python
# Import the MethodFactory

from photoholmes.methods.factory import MethodFactory

# Use the MethodFactory to import the method and preprocessing

method_name = "method_name"
config_path = "path_to_config"

method, preprocess = MethodFactory.load(method_name,config_path)

# Load an image

from photoholmes.utils.image import read_image

image_path = "image_path"
img = read_image(image_path)

# Preprocess the input and predict
image_data = {"image": image}
input_data = method_preprocessing(**image_data)
out = method.predict(**input_data)
```

## Create a method in the PhotoHolmes standard

### Create the method folder and source files

The main files are `method.py`, `preprocessing.py` and a configuration `config.yaml` file, but you may add any necessary modules such as neural network configurations or a `utils.py`. It may look something like this:

```
methods/
├── [other implemented methods]
└── your_method/
    ├── __init__.py
    ├── config.yaml
    ├── method.py
    ├── preprocessing.py
    ├── utils.py
    ├── models.py
    ├── [other relevant files]
    └── README.md
```

### Fill all of the corresponding files described in the Structure section

#### The method file

You must define your method as a child of the `BaseMethod` (or `BaseTorchMethod` if it is an end-to-end network), defining the behaviour of the class' methods `predict`, `benchmark` and  `__init__`. It should look something like this:

```python
from typing import Any, Tuple
from photoholmes.methods.base import BaseMethod, BenchmarkOutput
from torch import Tensor

from photoholmes.preprocessing.pipeline import PreProcessingPipeline

from .utils import example_function_1, example_function_2


class YourMethod(BaseMethod):

    def __init__(self, *params):
        super().__init__()
        # Attributes initialization ...

    def predict(self, image: Tensor, **kwargs) -> Tuple[Tensor, Tensor, float]:
        # Prediction pipeline
        features_1 = example_function_1(features_0, self.param1) #example
        # ...
        features_i = self.example_method_i(features_j)
        #...

        return heatmap, mask, detection

    def benchmark(self, image: Tensor) -> BenchmarkOutput:
        heatmap, mask, detection = self.predict(image)
        return {
            "heatmap": heatmap,
            "mask": mask,
            "detection": torch.tensor([detection]),
        }

    # ... Functions of interest ...
    def example_method_i(self, features_j: Tensor) -> Tensor:
        # ...
        return features_i
    # ...
```

#### The preprocessing file

A method's preprocessing can be described using the `PreProcessingPipeline`.
This is essentially a sequence of transforms, either imported from the preprocessing module or custom-defined, and input and output keys.

```python
from photoholmes.preprocessing.image import ExampleImagePreprocess
# Imports from other modules of "preprocessing"
from photoholmes.preprocessing.base import BasePreprocessing
from photoholmes.preprocessing.pipeline import PreProcessingPipeline

class ExampleCustomPreprocessing(BasePreprocessing):
    # CustomPreprocessing definition
    # ...
    # For details on how to create a custom preprocessing class, refer to the Preprocessing module documentation.

your_method_preprocessing = PreProcessingPipeline(
    inputs=["image"],
    outputs_keys=["outputs_of_interest"], # Coinciding with the predict method's keyword arguments.
    transforms=[
        ExampleImagePreprocessOne(),
        ExampleCustomPreprocess(),
        # More preprocess modules in the pipeline ...
    ],
)

```

#### A config file

The `config.yaml` file serves as a way to centralize customizable parameters of interest in a method.
 If the yaml file is organized appropiately, it can allow for the use of the `from_config` constructor method to create an instance of the method (which is also necessary if you wish to add the method to the factory and [contribute to the library](#readme_method-contribute)). In simple terms, it should suffice if the keys of the yaml file coincide with the input keyword arguments of the method's `__init__`.

The file can be organized in the following way:

```yaml
example_parameter_1: example_value1
example_parameter_2: example_value2
# More parameters ...

```

## <a id="readme_method-contribute"></a>Contribute: Adding a new method to the library

### Add to the MethodRegistry

Edit the file `src.methods.registry.py`.

```python
@unique
class MethodRegistry(Enum):
    # Previous Methods
    YOUR_METHOD = "your_method" # Maps a string identifier to an enumerate.

```

### Add to the MethodFactory

Edit the file `src.methods.factory.py`, inside the `load` method.

```python
match method_name:
    # Other method cases ...
    case MethodRegistry.YOUR_METHOD:
        from photoholmes.methods.your_method import (
            YourMethod, 
            your_method_preprocessing
            ) # Edit the __init__.py file if you wish to import in this way

        return TruFor.from_config(config), trufor_preprocessing
    
    case _:
    # Exception for not implemented cases
```

### Fill out a README file

Describe the main functionality of the method, give usage examples and add citations. Don't forget to include links to the weights if its a deep learning based method!

### Pull request to the repository

Make a pull request to the repository with the new method following the instructions of the [CONTRIBUTING.md](../../../CONTRIBUTING.md) file.
