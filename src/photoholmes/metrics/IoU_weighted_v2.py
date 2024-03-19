import torch
from torch import Tensor
from torchmetrics import Metric


class IoU_weighted_v2(Metric):
    """
    The IoU weighted (Intersection over Union weighted) metric calculates the IoU taking
    into account the value of the heatmap as a probability and uses weighted true
    positives, weighted false positives, weighted true negatives and weighted false
    negatives to calculate the IoU.
    This metric computes the IoU weighted score from the state of the metric. It
    accumulates True Positives, False Negatives and False Positives to then calculate
    the IoU weighted score.

    Attributes:
        TPw (Tensor): A tensor that accumulates the count of weighted true
        positives.
        FNw (Tensor): A tensor that accumulates the count of weighted false
        negatives.
        FPw (Tensor): A tensor that accumulates the count of weighted false
        positives.

    Methods:
        __init__(**kwargs): Initializes the IoU weighted metric object.
        update(preds: Tensor, target: Tensor): Updates the states with a new pair of
                                               prediction and target.
        compute() -> Tensor: Computes the IoU weighted from the state of the metric.

    Example:
        >>> IoU_weighted_metric = IoU_weighted_V2()
        >>> for preds, targets in data_loader:
        >>>     IoU_weighted_metric.update(preds, targets)
        >>> iou_weighted = IoU_weighted_metric.compute()
    """

    def __init__(self, **kwargs):
        """
        Initializes the IoU weighted metric object.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.add_state("TPw", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("FNw", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("FPw", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_images", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Updates the Tpw, Fpw and FNw counts with a new pair of
        prediction and target. It assumes both predictions as heatmap or binary
        and binary targets.

        Args:
            preds (Tensor): The predictions from the model.
                Expected to be a binary tensor or a heatmap.
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
        self.TPw += TPw
        self.FNw += FNw
        self.FPw += FPw
        self.total_images += torch.tensor(1)

    def compute(self) -> Tensor:
        """
        Computes the IoU weighted by using the accumulated TPw,
        FNw and FPw in the IoU weighted equation and then calculates the IoU weighted
        score.

        Returns:
            Tensor: The computed IoU weighted from the state of the metric.

        Note:
            If the total number of images is zero, it returns 0.0 to avoid division by
            zero.
        """
        if not self.total_images:
            return torch.tensor(0.0)
        TPw = self.TPw.float()
        FNw = self.FNw.float()
        FPw = self.FPw.float()
        denominator = TPw + FPw + FNw
        IoU_weighted = TPw / denominator if denominator != 0 else torch.tensor(0.0)
        return IoU_weighted
