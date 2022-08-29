"""The arg parsing class."""
import argparse
import typing


class Argparser:  # pylint: disable=too-few-public-methods
    """Argparser class."""

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument(
            "--equs",
            "-e",
            type=str,
            help="the path to the file containing the equations",
            default=False,
        )
        self.parser.add_argument(
            "--csv",
            "-c",
            type=str,
            help="the path to the CSV file containing the problem",
            default=False,
        )
        self.parser.add_argument(
            "--delim",
            "-l",
            type=str,
            help="the separator for the csv file",
            default=",",
        )
        self.parser.add_argument(
            "--slack",
            "-s",
            type=str,
            help="slack variable base name, names are creted"
            "by adding a number to the string",
            default="s",
        )
        self.parser.add_argument(
            "--aux",
            "-a",
            type=str,
            help="aux variable base name, names are creted"
            "by adding a number to the string",
            default="xa",
        )
        self.parser.add_argument(
            "--iter",
            "-i",
            type=int,
            help="maximum number of iterations",
            default=50,
        )
        self.parser.add_argument(
            "--min",
            "-m",
            action="store_true",
            help="determines whether its a minimization problem."
            "if not, its a maximization problem",
            default=False,
        )
        self.parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="whether to print output verbosely",
            default=False,
        )
        self.parser.add_argument(
            "--debug",
            "-d",
            action="store_true",
            help="whether to print debug info",
            default=False,
        )
        self.parser.add_argument(
            "--out",
            "-o",
            type=str,
            help="path to the output file",
            default="./lp_out.html",
        )
        # TODO- not being used right now
        self.parser.add_argument(
            "--numba",
            "-n",
            action="store_true",
            help="whether to print debug info",
            default=False,
        )
        self.args = self.parser.parse_args()

    def parse(self, args: typing.List[str]) -> None:
        """we only use this to parse the mock args from gui"""
        self.args = self.parser.parse_args(args)
