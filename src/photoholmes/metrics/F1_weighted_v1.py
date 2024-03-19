import torch
from torch import Tensor
from torchmetrics import Metric


class F1_weighted_v1(Metric):
    """
    The F1 weighted (F1 score weighted) metric calculates the F1 score taking
    into account the value of the heatmap as a probability and uses weighted true
    positives, weighted false positives, weighted true negatives and weighted false
    negatives to calculate the F1 score.
    This class computes de mean weighted F1. It calculates the weighted F1 score for
    each image and then averages to get the mean weighted F1 score.

    Attributes:
        F1_weighted (Tensor): A tensor that accumulates the count of F1
            score weighted across all the images.
        total_images (Tensor): A tensor that accumulates the count of images.

    Methods:
        __init__(**kwargs): Initializes the F1 score weighted metric object.
        update(preds: Tensor, target: Tensor): Updates the states with a new set of
            prediction and target.
        compute() -> Tensor: Computes the F1 score weighted from the state of the
            metric.

    Example:
        >>> F1_weighted_metric = F1_weighted_v1()
        >>> for preds, targets in data_loader:
        >>>     F1_weighted_metric.update(preds, targets)
        >>> f1_weighted = F1_weighted_metric.compute()
    """

    def __init__(self, **kwargs):
        """
        Initializes the F1 score weighted metric object.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.add_state("F1_weighted", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_images", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Updates the F1 score weighted counts with a new pair of
        prediction and target. It assumes both predictions as heatmap or binary
        and binary targets.

        Args:
            preds (Tensor): The predictions from the model. Expected to be a binary
                tensor or a heatmap.
            target (Tensor): The ground truth labels. Expected to be a binary tensor.

        Raises:
            ValueError: If the shapes of predictions and targets do not match.
        """
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        target = target.to(torch.int)
        pred_flat = preds.flatten()
        target_flat = target.flatten()
        TPw = torch.sum(pred_flat * target_flat)
        FPw = torch.sum((1 - pred_flat) * target_flat)
        FNw = torch.sum(pred_flat * (1 - target_flat))
        denominator = 2 * TPw + FNw + FPw
        if denominator != 0:
            self.F1_weighted += 2 * TPw / denominator
        self.total_images += torch.tensor(1)

    def compute(self) -> Tensor:
        """
        Computes the F1 weighted from the state of the metric.

        Returns:
            Tensor: The computed F1 weighted score.

        Note:
            If the total number of images is zero, it returns 0.0 to avoid division by
            zero.
        """
        f1_weighted = self.F1_weighted.float()
        total_images = self.total_images.float()
        return f1_weighted / total_images if total_images != 0 else torch.tensor(0.0)
