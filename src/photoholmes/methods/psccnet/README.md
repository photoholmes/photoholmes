# PSCC-Net: Progressive Spatio-Channel Correlation Network for Image Manipulation Detection and Localization

This is the implementation of the method by Liu et al. that can be found [here](https://arxiv.org/pdf/2103.10596.pdf).

The code contained in this library was derived from [the original implementation](https://github.com/proteus1991/PSCC-Net), making only minor changes to fit the PhotoHolmes library structure.

This is a deep learning based method, the weights can be found [here](https://github.com/proteus1991/PSCC-Net/tree/main/checkpoint) under the name [DetectionHead_checkpoint](https://github.com/proteus1991/PSCC-Net/tree/main/checkpoint/DetectionHead_checkpoint), [HRNet_checkpoint](https://github.com/proteus1991/PSCC-Net/tree/main/checkpoint/HRNet_checkpoint) and [NLCDetection_checkpoint](https://github.com/proteus1991/PSCC-Net/tree/main/checkpoint/NLCDetection_checkpoint). We last checked this information on March 9th 2024, please refer to the authors of the original paper if the weights cannot be found.

To easily download the weights, you can use the script in `scripts/download_psccnet_weights.py`.

## Description

PSCC-Net is an end-to-end fully convolutional neural network. It consists of a neural network that using a coarse to fine approach returns a mask locating forgeries in the input image. The method also returns an answer to the detection problem by returning a label that indicates whether the image was manipulated or not.

## Full overview

The network is divided into two different steps, first the top down path is constituted by a backbone
called HRNetV2p-W18. The main goal of this part is compute features at different scales that serve as inputs to the same levels of the bottom up path. The features obtained at every level of the top down path are used as inputs to a detection head that indicates if the image is pristine or not.

Then, the authors use in every level Spatio-Channel Correlation Module (SCCM) that tries to lay hold of spatial and channel wise correlations. Here, a coarse to fine approach is used, it involves an increasingly more precise definition of the masks as its shown in the bottom up path. The full architecture is trained on synthetic dataset that includes splicing, removal, copy move and pristine images.

## Usage

```python
from photoholmes.methods.psccnet import PSCCNet, psccnet_preprocessing

# Read an image
from photoholmes.utils.image import read_image
path_to_image = "path_to_image"
image = read_image(path_to_image)

# Assign the image to a dictionary and preprocess the image
image_data = {"image": image}
input = psccnet_preprocessing(**image_data)

# Declare the method and use the .to_device if you want to run it on cuda or mps instead of cpu
arch_config = "pretrained"
path_to_weights = {
        "FENet": "path_to_HRNet_weights",
        "SegNet": "path_to_NLCDetection_weights",
        "ClsNet": "path_to_DetectionHead_weights",
    }
method = PSCCNet(
    arch_config = arch_config,
    weights_paths = path_to_weights,
)
device = "cpu"
method.to_device(device)

# Use predict to get the final result
output = method.predict(**input)
```

## Citation

``` bibtex
@article{liu2022pscc,
  title={PSCC-Net: Progressive spatio-channel correlation network for image manipulation detection and localization},
  author={Liu, Xiaohong and Liu, Yaojie and Chen, Jun and Liu, Xiaoming},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2022},
  publisher={IEEE}
}
```
