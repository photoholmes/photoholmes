from typing import List, Union

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional.classification.auroc import binary_auroc


class mAuroc(Metric):
    """
    Compute the mean Area Under the Receiver Operating Characteristic curve (mAuroc).
    It accumulates de AUROC of every image and then outputs the mean value.

    Attributes:
        auroc (Tensor): A tensor that accumulates the the auroc values of each
                                image.
        total (Tensor): A tensor that accumulates the count of images.

    Methods:
        __init__(**kwargs): Initializes the mAUROC  metric object.
        update(preds: Tensor, target: Tensor): Updates the states with a new pair of
                                               prediction and target.
        compute() -> Tensor: Computes the mAUROC from the state of the metric.

    Example:
        >>> meanAUROC_metric = mAuroc()
        >>> for preds, targets in data_loader:
        >>>     meanAUROC_metric.update(preds, targets)
        >>> meanAUROC = meanAUROC_metric.compute()

    Note:
        The mAuroc is defined when the target tensor contains both positive and
        negative examples. If the target tensor is all zeros or all ones, the
        metric will be zero.
    """

    def __init__(self, thresholds: Union[int, List[float], None] = None, **kwargs):
        """
        Initializes the meanAUROC metric object.

        Args:
            thresholds: List of float values to use as thresholds for binarizing the
            heatmap. If None the default list of thresholds of AUROC is used.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)

        self.thresholds = thresholds
        self.add_state("auroc", default=torch.tensor(0.0))
        self.add_state("total", default=torch.tensor(0.0))

    def update(self, preds: Tensor, target: Tensor):
        """
        Updates the meanAUROC counts with a new pair of
        prediction and target. It assumes predictions as heatmap and binary targets.

        Args:
            preds (Tensor): The predictions from the model.
                Expected to be a heatmap.
            target (Tensor): The ground truth labels. Expected to be a binary tensor.

        Raises:
            ValueError: If the shapes of predictions and targets do not match.
        """
        bauroc = binary_auroc(preds, target, thresholds=self.thresholds)
        self.auroc += bauroc
        self.total += 1

    def compute(self):
        """
        Computes the meanAUROC over all the batches averaging the AUROCs of each image.

        Returns:
            Tensor: The computed meanAUROC from the state of the metric.

        Note:
            If the total number of images is zero, it returns 0.0 to avoid division by
            zero.
        """
        return self.auroc / self.total if self.total != 0 else torch.tensor(0.0)
