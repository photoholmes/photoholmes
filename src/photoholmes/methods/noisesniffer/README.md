# Noisesniffer: Image forgery detection based on noise inspection: analysis and refinement of the Noisesniffer method

This is the implementation of the method by Gardella et al. that can be found [here](https://www.ipol.im/pub/art/2024/462/).

The code contained in this library was derived from [the original implementation](https://ipolcore.ipol.im/demo/clientApp/demo.html?id=462), making only minor changes to fit the PhotoHolmes library structure.

## Description

The method exploits the consequences that the acquisition pipeline have on the noise model of a digital image. It estimates an stochastic model for said noise and detects noise anomalies using an a-contrario approach evaluating the number of false alarms (NFA). In order to get the suspected region of the forgery the authors of Noisesniffer use a region growing algorithm that detects the anomalous region according to the evaluation of the NFA.

## Full overview

The input images is divided in overlapping blocks of size $N \times N$ and the blocks that contain saturated pixels are discarded in order to avoid unreliable noise estimations. Then, for each channel, blocks are grouped in bins of a fixed size according to their mean intensity. Afterwards a channel is chosen and in that channel a bin is chosen to be processed according to the following steps: first, for each block of each bin, the DCT type II is computed, for each block the variance is computed using the low and medium frequencies of the DCT. In each bin, blocks are arranged in an ascending order according to the variance computed in the previous step and only a percentile of them is kept. The blocks kept in that step are the most homogeneous ones. After that, the variances of each of those blocks are computed and the blocks are ordered in an ascending order from which only the ones with the lowest variance are kept. If more than a certain amount of blocks have zero variance the bin is declared invalid. At the end, this part of the method obtains $L$, a group of blocks that are the most homogeneous and $V$ conformed by the blocks of $L$ that have the lowest variance.

The authors exploit the fact that the variance of the most homogeneous regions is explained by noise, which as a consequence,in the absence of tampering means that the spatial distribution of blocks of $V$ and $L$ should be the same, however some fluctuations in that spatial distribution are bound to happen due to randomness, the question that needs to be answered is whether the spatial distribution of blocks can be observed by chance or not. If the answer is the latter, that region could have been tampered with.

In order establish which regions are forgeries a region growing algorithm is used. This algorithm starts from a square tessellation of the image. Firstly, the authors describe a criterion in order to establish which of those cells could be meaningful, those cells are used as the initialization of the algorithm. From those cells, the algorithm iteratively adds contiguous cells that make the region more meaningful in the sense of the NFA.

## Usage

```python
from photoholmes.methods.noisesniffer import Noisesniffer, noisesniffer_preprocessing

# Read an image
from photoholmes.utils.image import read_image
path_to_image = "path_to_image"
image = read_image(path_to_image)

# Assign the image to a dictionary and preprocess the image
image_data = {"image": image}
input = noisesniffer_preprocessing(**image_data)

# Declare the method 
method = Noisesniffer()

# Use predict to get the final result
output = method.predict(**input)
```

## Citation

``` bibtex
@article{Noisesniffer,
  title={Image Forgery Detection Based on Noise Inspection: Analysis and Refinement of the Noisesniffer Method},
  author={Gardella, Marina and Mus{\'e}, Pablo and Colom, Miguel and Morel, Jean-Michel},
  journal={Preprint},
  year={2023},
  month={March},
  institution={Universitè Paris-Saclay, ENS Paris-Saclay, Centre Borelli, F-91190 Gif-sur-Yvette, France; IIE, Facultad de Ingenieria, Universidad de la Republica, Uruguay},
}
```
