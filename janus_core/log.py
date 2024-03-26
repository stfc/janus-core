"""Configure logger with yaml-styled format."""

import logging
from typing import Literal, Optional

from janus_core.janus_types import LogLevel

FORMAT = """
- timestamp: %(asctime)s
  level: %(levelname)s
  message: %(message)s
  trace: %(module)s
  line: %(lineno)d
""".strip()


def config_logger(
    name: str,
    filename: Optional[str] = None,
    level: LogLevel = logging.INFO,
    capture_warnings: bool = True,
    filemode: Literal["r", "w", "a", "x", "r+", "w+", "a+", "x+"] = "w",
    force: bool = False,
):
    """
    Configure logger with yaml-styled format.

    Parameters
    ----------
    name : str
        Name of logger. Default is None.
    filename : Optional[str]
        Name of log file if writing logs. Default is None.
    level : LogLevel
        Threshold for logger. Default is logging.INFO.
    capture_warnings : bool
        Whether to capture warnings in log. Default is True.
    filemode : str
        Mode of file to open. Default is "w".
    force : bool
        If true, remove and close existing handlers attached to the root logger before
        configuration. Default is False.

    Returns
    -------
    Optional[logging.Logger]
        Configured logger if `filename` has been specified.
    """
    if filename:
        logger = logging.getLogger(name)

        logging.basicConfig(
            level=level,
            filename=filename,
            filemode=filemode,
            format=FORMAT,
            encoding="utf-8",
            force=force,
        )
        logging.captureWarnings(capture=capture_warnings)
    else:
        logger = None
    return logger
