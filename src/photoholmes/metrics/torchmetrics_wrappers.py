from typing import List, Optional, Union

from torch import Tensor
from torchmetrics import Metric
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryMatthewsCorrCoef,
    BinaryPrecision,
    BinaryRecall,
    BinaryROC,
)


class AUROC(Metric):
    def __new__(
        cls, thresholds: Optional[Union[int, List[float], Tensor]] = None, **kwargs
    ):
        return BinaryAUROC(thresholds=thresholds, **kwargs)


class IoU(Metric):
    def __new__(cls, **kwargs) -> BinaryJaccardIndex:
        return BinaryJaccardIndex(**kwargs)


class MCC(Metric):
    def __new__(cls, **kwargs) -> BinaryMatthewsCorrCoef:
        return BinaryMatthewsCorrCoef(**kwargs)


class Precision(Metric):
    def __new__(cls, **kwargs):
        return BinaryPrecision(**kwargs)


class TPR(Metric):
    def __new__(cls, **kwargs):
        return BinaryRecall(**kwargs)


class F1Score(Metric):
    def __new__(cls, **kwargs):
        return BinaryF1Score(**kwargs)


class ROC(Metric):
    def __new__(cls, **kwargs):
        return BinaryROC(**kwargs)
