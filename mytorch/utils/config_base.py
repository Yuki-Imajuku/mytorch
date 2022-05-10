import argparse
from typing import List, Optional


class ConfigBase(object):
    """

    Base class for receiving learning conditions from command line arguments

    Attributes:
        parser (argparse.ArgumentParser): argument parser

    """

    def __init__(
        self,
        prog: Optional[str] = None,
        description: Optional[str] = None
    ) -> None:
        """

        __init__ function for `ConfigBase` class

        Args:
            prog (str, optional): program name showed in `--help` option
            description (str, optional): description showed in `--help` option

        Returns:
            None

        Examples:
            >>> def Config(ConfigBase):
            >>>     def __init__(self, prog: str, description: str) -> None:
            >>>         super().__init__(prog, description)
            >>>         self.parser.add_argument("--hoge")

        """
        self.parser = argparse.ArgumentParser(
            prog=prog,
            description=description
        )
        self.parser.add_argument(
            "input",
            help="input data directory"
        )
        self.parser.add_argument(
            "output",
            help="output model directory"
        )
        self.parser.add_argument(
            "log",
            help="output log directory"
        )
        self.parser.add_argument(
            "-g", "--gpus",
            nargs="*",
            type=int,
            help="using gpu number"
        )
        self.parser.add_argument(
            "-bs", "--batch-size",
            type=int,
            default=1024,
            help="batch size"
        )
        self.parser.add_argument(
            "-lr", "--learning-rate",
            type=float,
            default=0.001,
            help="learning rate"
        )
        self.parser.add_argument(
            "-ne", "--num-epochs",
            type=int,
            default=20,
            help="number of epochs"
        )
        self.parser.add_argument(
            "-v", "--verbose",
            action="store_true",
            help="show progress"
        )
        self.parser.add_argument(
            "-r", "--rich",
            action="store_true",
            help="show rich progress"
        )

    def read(
        self,
        args: Optional[List[str]]
    ) -> argparse.Namespace:
        """

        read and parse argument

        Args:
            args (list[str], optional): arguments, read stdin if args is None

        Returns:
            argparse.Namespace

        Examples:
            >>> config = ConfigBase()
            >>> config_from_stdin = config.read()
            >>> config_from_list = config.read(['./data', './models', './logs', '-lr', '0.1', '--gpus', '2', '3'])

        """
        return self.parser.parse_args(args)
