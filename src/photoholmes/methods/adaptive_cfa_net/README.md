# Adaptive CFA Net: An Adaptive Neural Network for Unsupervised Mosaic Consistency Analysis in Image Forensics

This is the implemenation of the method by Bammey et al. that can be found [here](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bammey_An_Adaptive_Neural_Network_for_Unsupervised_Mosaic_Consistency_Analysis_in_CVPR_2020_paper.pdf).

The code contained in this library was derived from [the original implementation](https://github.com/qbammey/adaptive_cfa_forensics), making only minor changes to fit the PhotoHolmes library structure.

This is a deep learning based method, the weights can be found [here](https://github.com/qbammey/adaptive_cfa_forensics/tree/master/src/models) under the name [pretrained.pt](https://github.com/qbammey/adaptive_cfa_forensics/blob/master/src/models/pretrained.pt) for the pretrained weights and [adapted_to_j95_database.pt](https://github.com/qbammey/adaptive_cfa_forensics/blob/master/src/models/adapted_to_j95_database.pt) for weights obtained through training with jpeg images. We last checked this information on March 9th 2024, please refer to the authors of the original paper if the weights cannot be found.

## Description

This research paper presents an innovative approach to automatically detect suspicious regions in potentially forged images. The method uses a Convolutional Neural Network (CNN) to identify inconsistencies in image mosaics, specifically targeting the artifacts left by demosaicing algorithms. Unlike many blind detection neural networks, this approach does not require labeled training data and can adapt to new, unseen data quickly.

## Full Overview

This research addresses the critical challenge of detecting image forgeries, focusing on the detection of demosaicing artifacts. Demosaicing is a key process in digital photography, where cameras use a Color Filter Array (CFA) to create color images, with the Bayer matrix being the most common. This process involves interpolating the full color image from pixels that are individually sampled in only one color, leaving unique artifacts. The study introduces a specialized Convolutional Neural Network (CNN) designed to detect these mosaic artifacts. Unlike traditional methods requiring extensive labeled datasets, this CNN can be trained on unlabelled, potentially forged images, showcasing an innovative approach in forensic image analysis.

To evaluate this method, the authors created a diverse benchmark database using the Dresden Image Database, processed with various demosaicing algorithms. This database comprises both authentic and forged images, where forgeries are created by splicing parts of images demosaiced differently. This setup allows for a detailed assessment of the network's capacity to detect inconsistencies in mosaic patterns indicative of forgery. The study demonstrates the network's effectiveness in detecting forgeries and its adaptability to different data types and compression formats, making a significant contribution to the field of image forensics by providing a robust, adaptable tool for unsupervised forgery detection.

## Usage

```python
from photoholmes.methods.adaptive_cfa_net import (
    AdaptiveCFANet,
    adaptive_cfa_net_preprocessing,
)

# Read an image
from photoholmes.utils.image import read_image
path_to_image = "path_to_image"
image = read_image(path_to_image)

# Assign the image to a dictionary and preprocess the image
image_data = {"image": image}
input = adaptive_cfa_net_preprocessing(**image_data)

# Declare the method and use the .to_device if you want to run it on cuda or mps instead of cpu
arch_config = "pretrained"
path_to_weights = "path_to_weights"
method = AdaptiveCFANet(
    arch_config=arch_config,
    weights=path_to_weights,
)
device = "cpu"
method.to_device(device)

# Use predict to get the final result
output = method.predict(**input)
```

## Citation

``` bibtex
@InProceedings{Bammey_2020_CVPR,
author = {Bammey, Quentin and Gioi, Rafael Grompone von and Morel, Jean-Michel},
title = {An Adaptive Neural Network for Unsupervised Mosaic Consistency Analysis in Image Forensics},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```
