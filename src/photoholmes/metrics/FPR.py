import torch
from torch import Tensor
from torchmetrics import Metric


class FPR(Metric):
    """
    The FPR (False Positive Rate) metric calculates the proportion of
    false positive predictions in relation to the total number of actual
    negative samples.

    Attributes:
        false_positives (Tensor): A tensor that accumulates the count of false
            positive predictions in a single image.
        total_negatives (Tensor): A tensor that accumulates the count of true
            negative instances in a single image.

    Methods:
        __init__(**kwargs): Initializes the FPR metric object.
        update(preds: Tensor, target: Tensor): Updates the states with a new pair of
            prediction and target.
        compute() -> Tensor: Computes the False Positive Rate from the state of the
            metric.

    Example:
        >>> fpr_metric = FPR()
        >>> for preds, targets in data_loader:
        >>>     fpr_metric.update(preds, targets)
        >>> fpr = fpr_metric.compute()
    """

    def __init__(self, **kwargs):
        """
        Initializes the FPR metric object.

        Args:
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.add_state("false_positives", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_negatives", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """
        Updates the false positives and total negatives counts with a new pair of
        prediction and target. It assumes both predictions and targets are binary.

        Args:
            preds (Tensor): The predictions from the model.
                Expected to be a binary tensor.
            target (Tensor): The ground truth labels. Expected to be a binary tensor.

        Raises:
            ValueError: If the shapes of predictions and targets do not match.
        """
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        self.false_positives += torch.sum((preds == 1) & (target == 0))
        self.total_negatives += torch.sum(target == 0)

    def compute(self) -> Tensor:
        """
        Computes the False Positive Rate from the state of the metric by using the
        accumulared false_positives and total_negatives.

        Returns:
            Tensor: The computed False Positive Rate.

        Note:
            If the total number of negatives is zero, it returns 0.0 to avoid division
            by zero.

        """
        false_positives = self.false_positives.float()
        total_negatives = self.total_negatives.float()
        return (
            false_positives / total_negatives
            if total_negatives != 0
            else torch.tensor(0.0)
        )
