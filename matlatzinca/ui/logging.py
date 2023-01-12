import logging
from pathlib import Path


def initialize_logger(logfile=None, level=logging.INFO, console=False):

    handlers = []
    if logfile is not None and Path(logfile).exists():
        handlers.append(logging.FileHandler(logfile, mode="w"))
    if console:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)s: %(message)s (%(name)s)",
        datefmt="%H:%M:%S",
        level=level,
        handlers=handlers,
    )

    logging.info("Starting Matlatzinca log.")
    return logging
