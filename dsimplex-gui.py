#!/usr/bin/env python3
"""The GUI frontend."""

from dsimplex import gui


def main():
    """Entrypoint for the GUI."""
    dsimplex_gui = gui.DsimplexGui()
    dsimplex_gui.main_loop()


if __name__ == "__main__":
    main()
