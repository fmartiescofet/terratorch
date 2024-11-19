# Copyright contributors to the Terratorch project

"""Command-line interface to TerraTorch."""

from terratorch.cli_tools import build_lightning_cli
from terratorch.my_profiler import PROFILER


def main():
    with PROFILER:
        _ = build_lightning_cli()


if __name__ == "__main__":
    main()
