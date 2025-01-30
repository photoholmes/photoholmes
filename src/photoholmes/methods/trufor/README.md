# TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization

This is the implementation of the method by Guillaro et al. that can be found [here](https://arxiv.org/pdf/2212.10957.pdf).

The code contained in this library was derived from [the original implementation](https://github.com/grip-unina/TruFor), making only minor changes to fit the PhotoHolmes library structure.

This is a deep learning based method, the weights can be found [here](https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip). We last checked this information on March 9th 2024, please refer to the authors of the original paper if the weights cannot be found.

## Description

The paper presents a novel approach to detect and localize image forgeries. The method
extracts both high-level and low-level features through a transformer-based architecture
that combines the RGB image and a learned noise-sensitive fingerprint. The latter one
is a re-train of the [noiseprint method](https://ieeexplore.ieee.org/document/8713484),
dubbed noiseprint++. The forgeries are detected as deviations from the expected regular pattern
that characterizes a pristine image.

On top of a pixel-level localization map and a whole-image integrity score, the method outputs
a reliability map that highlights areas where the localization predictions may be error-prone, reducing false-alarms.

## Full overview

The full TruFor framework consists of an RGB image is the input of the framework from which a noise feature is extracted with the NoisePrint++ module. Both, the RGB image and the noise feature, are the inputs of an encoder that extracts dense features used in the next steps of the framework. Those features serve as input to an anomaly decoder from which an anomaly map is extracted. This anomaly map is the heat map that allows users to identify forged regions. Those same features are also an input of a confidence decoder whose output is the confidence map. The pipeline also combines the output of both decoders with a pooling module in order to get a compact descriptor that goes through a forgery detector module in order to predict an integrity score necessary in order to give an answer to the detection problem.

The NoisePrint++ module is constituted of the same architecture used to extract the NoisePrint. The difference is that it was retrained with a wider variety of pristine images. The encoder is based on a CMX architecturethat relies on SegFormer modules. Both decoders have the architecture of the multilayer perceptron of SegFormer and were trained with a dataset including pristine and forged images with their respective ground truth masks. In the case of the confidence decoder the true class probability map was used as a mask in order to capture the confidence of the anomaly prediction. The pooling module is in charge of generating an 8-component feature vector that is fed to two fully connected layers that predict the integrity score. Those fully connected layers that conform the forgery detector are trained with the same dataset as the decoders.

## Usage

```python
from photoholmes.methods.trufor import TruFor, trufor_preprocessing

# Read an image
from photoholmes.utils.image import read_image
path_to_image = "path_to_image"
image = read_image(path_to_image)

# Assign the image to a dictionary and preprocess the image
image_data = {"image": image}
input = trufor_preprocessing(**image_data)

# Declare the method and use the .to_device if you want to run it on cuda or mps instead of cpu
path_to_weights = "path_to_weights"
method = TruFor(
    weights= path_to_weights
)
device = "cpu"
method.to_device(device)

# Use predict to get the final result
output = method.predict(**input)
```

## Citation

``` bibtex
@misc{guillaro2023trufor,
      title={TruFor: Leveraging all-round clues for trustworthy image forgery detection and localization}, 
      author={Fabrizio Guillaro and Davide Cozzolino and Avneesh Sud and Nicholas Dufour and Luisa Verdoliva},
      year={2023},
      eprint={2212.10957},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
