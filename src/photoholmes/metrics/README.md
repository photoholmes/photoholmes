# Metrics Module

## Table of Contents

- [Metrics Module](#metrics-module)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Available Metrics](#available-metrics)
    - [Torch-metrics:](#torch-metrics)
    - [Custom metrics:](#custom-metrics)
  - [Metric Factory](#metric-factory)
  - [Examples of Use](#examples-of-use)
    - [Using a single metric:](#using-a-single-metric)
    - [Using the MetricFactory:](#using-the-metricfactory)
  - [Contribute: Adding a new metric](#contribute-adding-a-new-metric)

## Overview

This module provides a collection of metrics for evaluating the performance of a method for image forgery detection and localization.

The metrics are divided into two categories:

- Metrics imported from [torch-metrics](https://lightning.ai/docs/torchmetrics/stable/).
- [Custom metrics](custom_metrics.md) using torch-metrics.

## Available Metrics

### Torch-metrics:

- `AUROC`: Area Under the Receiver Operating Characteristic curve. [Docs](https://lightning.ai/docs/torchmetrics/stable/classification/auroc.html) 
- `IoU`: Intersection over Union, also known as Jaccard Index. [Docs](https://lightning.ai/docs/torchmetrics/stable/classification/jaccard_index.html)
- `MCC`: Matthews Correlation Coefficient. [Docs](https://lightning.ai/docs/torchmetrics/stable/classification/matthews_corr_coef.html)
- `Precision`: [Docs](https://lightning.ai/docs/torchmetrics/stable/classification/precision.html)
- `ROC`: Receiver Operating Characteristic curve. [Docs](https://lightning.ai/docs/torchmetrics/stable/classification/roc.html)
- `F1 score`: [Docs](https://lightning.ai/docs/torchmetrics/stable/classification/f1_score.html)
- `TPR`: True Positive Rate or recall. [Docs](https://lightning.ai/docs/torchmetrics/stable/classification/recall.html)

For more information about the available metrics in torch-metrics please refer to the [documentation](https://lightning.ai/docs/torchmetrics/stable/).

### Custom metrics:

- `FPR`: False Positive Rate.
- `meanAUROC`: Mean Area Under the Receiver Operating Characteristic curve.
- `IoU_weighted_v1`: Weighted Intersection over Union v1.
- `IoU_weighted_v2`: Weighted Intersection over Union v2.
- `MCC_weighted_v1`: Weighted Mathews Correlation Coefficient v1.
- `MCC_weighted_v2`: Weighted Mathews Correlation Coefficient v2.
- `F1_weighted_v1`: Weighted F1 score v1.
- `F1_weighted_v2`: Weighted F1 score v2.

For more information about the custom metrics please refer to the [documentation](custom_metrics.md).

## Metric Factory

The `MetricFactory` class provides a way to load multiple metrics at once. It is useful when you need to evaluate the performance of a method using multiple metrics.

It returns a [MetricCollection](https://lightning.ai/docs/torchmetrics/stable/pages/overview.html#metriccollection) object that contains all the metrics loaded.

## Examples of Use

Here are some examples of how to use the metrics in this module:

### Using a single metric:

You can use the metrics directly by instantiating the class and passing the predictions and the ground truth masks as arguments. Here is an example using the `AUROC` metric:

```python
from photoholmes.metrics import AUROC

import torch

# generate a random mask of size 256x256
mask = torch.randint(0, 2, (256, 256))
# generate a random prediction of probabilities of size 256x256 
pred = torch.rand(256, 256)
# instantiate the metric
auroc_metric = AUROC()
# calculate the metric
auroc = auroc_metric(pred, mask)

print("AUROC:", auroc)
```

For computing the metrics over several images, call the `update` method to update the metric with the predictions and the ground truth masks. Then, call the `compute` method to get the value of the metric. Here is an example using the `AUROC` metric:

```python
from photoholmes.metrics import AUROC

import torch

auroc_metric = AUROC()

# Generate random data
data = [(torch.rand(256, 256), torch.randint(0, 2, (256, 256))) for _ in range(10)]

for pred, mask in data:
    auroc_metric.update(pred, mask)
auroc = auroc_metric.compute()

print("AUROC:", auroc)
```

Using the custom metrics is the same as using the metrics from
torch-metrics. Here is an example using the `IoU_weighted_v1` metric:

```python
from photoholmes.metrics import IoU_weighted_v1

import torch

iou_weighted_v1_metric = IoU_weighted_v1()

# Generate random data
data = [(torch.rand(256, 256), torch.randint(0, 2, (256, 256))) for _ in range(10)]

for pred, mask in data:
    iou_weighted_v1_metric.update(pred, mask)
iou_weighted_v1 = iou_weighted_v1_metric.compute()

print("IoU_weighted_v1:", iou_weighted_v1)
```

### Using the MetricFactory:

Metrics can be loaded by passing a list of metric names to the `load` method. Here is an example using the `MetricFactory` to load the `MCC` and `F1_weighted_v2` metrics:

```python
from photoholmes.metrics import MetricFactory

import torch

metric = MetricFactory.load(["mcc", "f1_weighted_v2"])

# Generate random data
data = [(torch.rand(256, 256), torch.randint(0, 2, (256, 256))) for _ in range(10)]

for pred, mask in data:
    metric.update(pred, mask)
metric_value = metric.compute()

print(metric_value)
```

Metrics can also be loaded by passing a list of `MetricRegistry` objects to the `load` method. Here is an example using the `MetricFactory` to load the `mAUROC` metric:

```python
from photoholmes.metrics import MetricFactory, MetricRegistry

import torch

# Generate random data
data = [(torch.rand(256, 256), torch.randint(0, 2, (256, 256))) for _ in range(10)]

metric_collection = MetricFactory.load([MetricRegistry.mAUROC])

for pred, mask in data:
    metric_collection.update(pred, mask)
metric_value = metric_collection.compute()

print(metric_value)
```

With the `MetricRegistry` class you can get all the available metrics and load them all at once. Here is an example using the `MetricFactory` to load all the available metrics:

```python
from photoholmes.metrics import MetricFactory, MetricRegistry

import torch

# Generate random data
data = [(torch.rand(256, 256), torch.randint(0, 2, (256, 256))) for _ in range(10)]

metric_names = MetricRegistry.get_all_metrics()
metric_collection = MetricFactory.load(metric_names)

for pred, mask in data:
    metric_collection.update(pred, mask)
metric_value = metric_collection.compute()

print(metric_value)
```

To add a metric to the `MetricCollection`, you can use the `add` method of the `MetricCollection` object. For more information about the `MetricCollection` object please refer to the [documentation](https://lightning.ai/docs/torchmetrics/stable/pages/overview.html#metriccollection). Here is an example using the `MetricCollection` to add the `AUROC` metric:

```python
from photoholmes.metrics import MetricFactory, MetricRegistry
import torch
from torchmetrics import Precision

# Generate random data
data = [(torch.rand(256, 256), torch.randint(0, 2, (256, 256))) for _ in range(10)]

metric_collection = MetricFactory.load([MetricRegistry.AUROC])
metric_collection.add_metrics(Precision(task="binary"))

for pred, mask in data:
    metric_collection.update(pred, mask)
metric_value = metric_collection.compute()

print(metric_value)
```

## Contribute: Adding a new metric

If the metric already exists in [torch-metrics](https://lightning.ai/docs/torchmetrics/stable/) the steps to follow are:

1. Add metric to registry

    ```python
    class MetricRegistry(Enum):
        NEW_TORCHMETRIC = "new_torch_metric"
    ```

2. Add a wrapper for the metric in the `torchmetrics_wrapper.py` file if necessary. This file contains the wrapper for the metrics from [torch-metrics](https://lightning.ai/docs/torchmetrics/stable/). The wrapper should follow the same pattern as the other metrics in the file. It is implemented as a class that inherits from the `Metric` class, and has the same signature as the original.
3. Add the metric to the `__init__.py` file of the metrics module.
4. Add the metric to the factory by following this template

    ``` python
    case MetricRegistry.NEW_TORCHMETRIC:
        from photoholmes.metrics import New_Torch_Metric as NTM
        metrics.append(NTM())
    ```

If the metric does not exist in [torch-metrics](https://lightning.ai/docs/torchmetrics/stable/) you should follow the instructions provided by [torch-metrics](https://lightning.ai/docs/torchmetrics/stable/) [here](https://lightning.ai/docs/torchmetrics/stable/pages/implement.html), so the steps are as follows:

1. Create the .py file as explained in the tutorial of [torch-metrics](https://lightning.ai/docs/torchmetrics/stable/).
2. Add metric to registry

    ```python
    class MetricRegistry(Enum):
        FANCY_NEW_METRIC = "fancy_new_metric"
    ```

3. Add the metric to the `__init__.py` file of the metrics module.
4. Add the metric to the factory by following this template

    ``` python
    case MetricRegistry.FANCY_NEW_METRIC:
        from photoholmes.metrics import Fancy_New_Metric as FNM
        metrics.append(FNM())
    ```

Make a pull request to the repository with the new metric following the instructions of the [CONTRIBUTING.md](../../../CONTRIBUTING.md) file.
