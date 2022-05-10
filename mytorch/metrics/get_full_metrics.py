from typing import Optional
from torchmetrics import F1Score, Accuracy, MetricCollection, Precision, Recall


def get_full_metrics(
    num_classes: Optional[int] = None,
    threshold: float = 0.5,
    average: str = "macro",
    ignore_index: Optional[int] = None,
    top_k: Optional[int] = None,
    multiclass: Optional[bool] = None,
    prefix: Optional[str] = None
) -> MetricCollection:
    """

    return metric collection (Accuracy, Precision, Recall, F1)

    Args:
        num_classes (int, optional): number of class
        threshold (float, optional): threshold
        average (str, optional): average method
        ignore_index (int, optional): ignore index
        top_k (int, optional): save top k model
        multiclass (bool, optional): multiclass
        prefix (str, optional): prefix string

    Returns:
        MetricCollection: metric collection

    Examples:
        >>> train_metrics = get_full_metrics(prefix="train_")  # "train_Accuracy", ...
        >>> val_metrics = get_full_metrics(prefix="val_")      # "val_Accuracy", ...

    """
    return MetricCollection([
        Accuracy(
            threshold=threshold,
            num_classes=num_classes,
            average=average,
            ignore_index=ignore_index,
            top_k=top_k,
            multiclass=multiclass
        ),
        Precision(
            num_classes=num_classes,
            threshold=threshold,
            average=average,
            ignore_index=ignore_index,
            top_k=top_k,
            multiclass=multiclass
        ),
        Recall(
            num_classes=num_classes,
            threshold=threshold,
            average=average,
            ignore_index=ignore_index,
            top_k=top_k,
            multiclass=multiclass
        ),
        F1Score(
            num_classes=num_classes,
            threshold=threshold,
            average=average,
            ignore_index=ignore_index,
            top_k=top_k,
            multiclass=multiclass
        )
    ], prefix=prefix)
