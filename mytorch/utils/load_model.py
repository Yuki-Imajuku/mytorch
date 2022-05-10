from pathlib import Path
from typing import Any, Union


def load_model(model: Any, path: Union[str, Path]) -> Any:
    """

    load model from checkpoint file

    Args:
        model (Any): model
        path (str | pathlib.Path): file path

    Returns:
        Any: model

    Examples:
        >>> class NNModel(pl.LightningModule):
        :
        >>> model = load_model(NNModel(), "./models/last.ckpt")

    Note:
        `model` must be inherited `pytorch_lightning.LightningModule` class

    """
    if isinstance(path, str):
        model.load_from_checkpoint(path)
    elif isinstance(path, Path):
        model.load_from_checkpoint(str(path))
    else:
        raise ValueError("path (str | pathlib.Path)")
    return model
