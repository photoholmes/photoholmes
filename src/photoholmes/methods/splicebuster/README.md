# Splicebuster: a new blind image splicing detector

This is the implementation of the method by Cozzolino et al. that can be found [here](https://ieeexplore.ieee.org/abstract/document/7368565) paper.

The code contained in this library was derived from [the original implementation](https://www.grip.unina.it/download/prog/Splicebuster/) and from a code provided by Marina Gardella, Quentin Bammey and Tina Nikoukhah, making only minor changes to fit the PhotoHolmes library structure.

## Description

Splicebuster is a method based on finding anomalies in an image's residual, obtaining a set of features from the co-ocurrence matrix of a local estimation of the residual, which is assumed homogenous in the absence of forgery, and contain anomalies in the presence of such. This last assumption is explained through the fact that the image residual (from which the method's pipeline begins) is a reasonable estimation of the image's noise, which should have the same model in images of the same camera and different in other cases. This is also one of the reasons why it is selected for the PhotoHolmes project: it is a model-driven approach that offers interpretability, in contrast which most models which include some form of Deep Learning.

## Full overview

Firstly, it estimates the image's residual by high-pass filtering. More specifically, it is computed by estimating the third order derivatives in both vertical and horizontal directions, obtaining two residuals $r_v$ and $r_h$. These are quantized in $3$ levels, and co-ocurrence features are extracted from them. These features are dimentionally reduced, both by merging complementary and simetrical co-ocurrences, then taking histograms of these co-ocurrences, and finally applying PCA (Principal Component Analysis) of these historams. This results in each block having 2 histograms of 25 bins as features.

Over this feature map, an Expectation-Maximization algorithm is applied to estimate a two-class mixture model behind it. This can be done with two types of mixtures, either gaussian-uniform or gaussian-gaussian. The concept behind this is that the forged region can have either uniform or gaussian distribution depending on the type of forgery applied. The expectation step computes a conditional expected value of the log-likelihood, given a set of parameters. The maximization step finds the optimal set of parameters.

Finally, the heatmap is produced by studying the relation between the Mahalanobis distance of the block's feature histogram and each of the models.

## Usage

```python
from photoholmes.methods.splicebuster import Splicebuster, splicebuster_preprocessing

# Read an image
from photoholmes.utils.image import read_image
path_to_image = "path_to_image"
image = read_image(path_to_image)

# Assign the image to a dictionary and preprocess the image
image_data = {"image": image}
input = splicebuster_preprocessing(**image_data)

# Declare the method
method = Splicebuster()

# Use predict to get the final result
output = method.predict(**input)
```

## Citation

``` bibtex
@INPROCEEDINGS{Splicebuster,
  author={Cozzolino, Davide and Poggi, Giovanni and Verdoliva, Luisa},
  booktitle={2015 IEEE International Workshop on Information Forensics and Security (WIFS)}, 
  title={Splicebuster: A new blind image splicing detector}, 
  year={2015},
  pages={1-6},
  doi={10.1109/WIFS.2015.7368565}
}
```
