"""Console launcher, separate from the run's pinned inference implementation."""

def main(argv=None):
    from .cli import main as launch_main

    return launch_main(argv)
