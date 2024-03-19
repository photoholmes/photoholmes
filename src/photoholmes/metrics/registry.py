from enum import Enum, unique


@unique
class MetricRegistry(Enum):
    AUROC = "auroc"
    mAUROC = "mauroc"
    FPR = "fpr"
    IoU = "iou"
    MCC = "mcc"
    Precision = "precision"
    ROC = "roc"
    TPR = "tpr"
    IoU_WEIGHTED_V1 = "iou_weighted_v1"
    F1_WEIGHTED_V1 = "f1_weighted_v1"
    MCC_WEIGHTED_V1 = "mcc_weighted_v1"
    IoU_WEIGHTED_V2 = "iou_weighted_v2"
    F1_WEIGHTED_V2 = "f1_weighted_v2"
    MCC_WEIGHTED_V2 = "mcc_weighted_v2"
    F1 = "f1"

    @classmethod
    def get_all_metrics(cls):
        metric_names = list(MetricRegistry)
        metrics = [metric.value for metric in metric_names]
        return metrics
