

# Custom metrics

## Overview

This module contains custom metrics that are not available in [torch-metrics](https://lightning.ai/docs/torchmetrics/stable/). The metrics are designed to evaluate the performance of image forgery detection and localization methods.

## Metrics

### FPR
The False Positive Rate, is defined as the ratio of false positives to the sum of false positives and true negatives.
$$TPR = \frac{FP}{FP + TN}$$
### meanAUROC
Calculates de auroc for each image and then averages those values over the full dataset.
### Weighted metrics
These metrics allow the comparison of performance between methods whose outputs are masks with methods whose outputs are heatmaps by regarding said heatmaps as maps of probability in which the value of each pixel corresponds to the probability of the pixel being forged. The same works with the detection problem in which the output is a single number that indicates the probability of the image as whole being forged. To accomplish that [Gardella, 2023](https://ipolcore.ipol.im/demo/clientApp/demo.html?id=77777000341) and [Bammey, 2021](https://openaccess.thecvf.com/content/WACV2022/papers/Bammey_Non-Semantic_Evaluation_of_Image_Forensics_Tools_Methodology_and_Database_WACV_2022_paper.pdf) define 
weighted true positives, weighted false positives, weighted true negatives, and weighted false negatives as follows:

  $$TP_w = \sum_xH(x)M(x)$$

  $$FP_w = \sum_x(1-H(x))M(x)$$

  $$TN_w = \sum_x(1-H(x))(1-M(x))$$

  $$FN_w = \sum_xH(x)(1-M(x))$$
in which $H(x)$ corresponds to the predicted output and $M(x)$ corresponds to the mask.

The implemented weighted metrics are:
#### Weighted MCC: Weighted Mathews Correlation Coefficient
$$MCC_{weighted} = \frac{TP_w \times TN_w - FP_w \times  FN_w}{\sqrt{(TP_w + FP_w)(TP_w+FN_w)(TN_w+FP_W)(TN_w+FN_w)}}$$
#### Weighted IoU: Weighted Intersection over Union
$$IoU_{weighted} = \frac{TP_w}{TP_w + FN_w + FP_w}$$
#### Weighted F1: Weighted F1 score
$$F1_{weighted} = \frac{2TP_w}{2TP_w + FN_w + FP_w}$$
    
There are two versions of the weighted metrics:
- v1: Corresponds to the mean version of each weighted metric. Those metrics accumulate the value of the metric for each image and then the output is the average of the metric over the full dataset. 
This version of the metric is recommended to evaluate localization performance.
- v2: Corresponds to the value of the metric over the full dataset as defined in [torch-metrics](https://lightning.ai/docs/torchmetrics/stable/). For each image the metric accumulates the FPw, TPW, TNw and FNw and then with those accumulations outputs the value of the metric for the full dataset. 
This version of the metric is recommended to evaluate detection performance.

## References

```tex
@article{Noisesniffer,
  title={Image Forgery Detection Based on Noise Inspection: Analysis and Refinement of the Noisesniffer Method},
  author={Gardella, Marina and Mus{\'e}, Pablo and Colom, Miguel and Morel, Jean-Michel},
  journal={Preprint},
  year={2023},
  month={March},
  institution={Universitè Paris-Saclay, ENS Paris-Saclay, Centre Borelli, F-91190 Gif-sur-Yvette, France; IIE, Facultad de Ingenieria, Universidad de la Republica, Uruguay},
}
```

```tex
@misc{bammey2021nonsemantic,
      title={Non-Semantic Evaluation of Image Forensics Tools: Methodology and Database}, 
      author={Quentin Bammey and Tina Nikoukhah and Marina Gardella and Rafael Grompone and Miguel Colom and Jean-Michel Morel},
      year={2021},
      eprint={2105.02700},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```