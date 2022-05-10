from typing import Optional, Union

from pytorch_lightning.callbacks import (
    RichProgressBar,
    TQDMProgressBar
)
from pytorch_lightning.callbacks.progress.rich_progress import (
    RichProgressBarTheme
)


def progress_bar(
    rich: Optional[bool] = False
) -> Union[RichProgressBar, TQDMProgressBar]:
    """

    setup progress bar for Trainer

    Args:
        rich (bool): flag for progress bar type

    Returns:
        RichProgressBar | TQDMProgressBar: progress bar

    Examples:
        >>> tqdm_bar = progress_bar()      # TQDMProgressBar
        >>> rich_bar = progress_bar(True)  # RichProgressBar

    Note:
        requires `rich` module when rich = True

    """
    if rich:
        return RichProgressBar(
            theme=RichProgressBarTheme(
                description="green_yellow",
                progress_bar="green1",
                progress_bar_finished="green1",
                progress_bar_pulse="#6206E0",
                batch_progress="green_yellow",
                time="grey82",
                processing_speed="grey82",
                metrics="grey82",
            ),
            refresh_rate=1
        )
    else:
        return TQDMProgressBar(refresh_rate=1)
