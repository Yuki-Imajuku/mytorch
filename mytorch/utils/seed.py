import os
import random

import numpy as np
import torch


class Seed:
    """

    Class for setting random seed

    Attributes:
        os (bool): flag for `os` module
        random (bool): flag for `random` module
        numpy (bool): flag for `numpy` module
        torch (bool): flag for `torch` module

    """

    def __init__(
        self,
        _os: bool = True,
        _random: bool = True,
        _numpy: bool = True,
        _torch: bool = True
    ) -> None:
        """

        __init__ function for `Seed` class

        Args:
            _os (bool, optional): flag for `os` module
            _random (bool, optional): flag for `random` module
            _numpy (bool, optional): flag for `numpy` module
            _torch (bool, optional): flag for `torch` module

        Returns:
            None

        Examples:
            >>> seed_setter = Seed()

        """
        self.os = _os
        self.random = _random
        self.numpy = _numpy
        self.torch = _torch

    def set(
        self,
        seed: int
    ) -> None:
        """

        set seed

        Args:
            seed (int): random seed

        Returns:
            None

        Examples:
            >>> Seed().set(42)

        Note:
            `seed` must satisfy 0 <= `seed` < 2 ^ 31 if `self.numpy` = True

        """
        if self.os:
            os.environ["PYTHONHASHSEED"] = str(seed)
        if self.random:
            random.seed(seed)
        if self.numpy:
            np.random.seed(seed)
        if self.torch:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
