import torch
from torch import Tensor
from torchmetrics import Metric


class MCC_weighted_v2(Metric):
    """
    The MCC weighted (Mathews Correlation Coefficient weighted) metric calculates the
    MCC score taking into account the value of the heatmap as a probability and uses
    weighted true positives, weighted false positives, weighted true negatives and
    weighted false negatives to calculate the MCC score.
    This metric computes the MCC weighted score. It accumulates
    True Positives, False Negatives, False Positives and True Negatives to then
    calculate the MCC weighted score.

      Attributes:
        TPw (Tensor): A tensor that accumulates the count of weighted true
        positives.
        FNw (Tensor): A tensor that accumulates the count of weighted false
        negatives.
        FPw (Tensor): A tensor that accumulates the count of weighted false
        positives.
        TNw (Tensor): A tensor that accumulates the count of weighted true
        negatives.

    Methods:
        __init__(**kwargs): Initializes the MCC score weighted metric object.
        update(preds: Tensor, target: Tensor): Updates the states with a new pair of
                                               prediction and target.
        compute() -> Tensor: Computes the MCC score weighted from the state of the
            metric.

    Example:
        >>> MCC_weighted_metric = MCC_weighted_v2()
        >>> for preds, targets in data_loader:
        >>>     MCC_weighted_metric.update(preds, targets)
        >>> mcc_weighted = MCC_weighted_metric.compute()
    """

    def __init__(self, **kwargs):
        """
        Initializes the MCC weighted metric object.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.add_state("TPw", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("FNw", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("FPw", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("TNw", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_images", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Updates the TPw, FNw, FPw and TNw counts with a new pair of
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
        TNw = torch.sum((1 - pred_flat) * (1 - target_flat))
        FNw = torch.sum(pred_flat * (1 - target_flat))
        self.TPw += TPw
        self.FNw += FNw
        self.FPw += FPw
        self.TNw += TNw
        self.total_images += torch.tensor(1)

    def compute(self) -> Tensor:
        """
        Computes the MCC weighted by using the accumulated TPw, FNw, FPw and TNw in the
        MCC weighted equation.

        Returns:
            Tensor: The computed MCC weighted from the state of the metric.

        Note:
            If the total number of images is zero, it returns 0.0 to avoid division by
            zero.
        """
        if self.total_images == 0:
            return torch.tensor(0.0)
        TPw = self.TPw.float()
        FNw = self.FNw.float()
        FPw = self.FPw.float()
        TNw = self.TNw.float()
        if TPw + TNw != 0 and FPw + FNw == 0:
            return torch.tensor(1.0)
        if TPw + TNw == 0 and FPw + FNw != 0:
            return torch.tensor(-1.0)
        denominator = torch.sqrt((TPw + FPw) * (TPw + FNw) * (TNw + FPw) * (TNw + FNw))
        MCC_weighted = (
            (TPw * TNw - FPw * FNw) / denominator
            if denominator != 0
            else torch.tensor(0.0)
        )
        return MCC_weighted
