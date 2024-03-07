"""Configure logger."""

import logging

from janus_core.janus_types import LogType

FORMAT = """
- timestamp: %(asctime)s
  level: %(levelname)s
  message: %(message)s
  trace: %(module)s
  line: %(lineno)d
""".strip()


def config_logger(
    name: str,
    level: LogType = logging.INFO,
    filename: str = "janus.log",
    capture_warnings: bool = True,
):
    """
    Configure logger.

    Parameters
    ----------
    name : str
        Name of logger.
    level : LogType
        Threshold for logger. Default is logging.INFO.
    filename : str
        Name of log file to write logs to. Default is "janus.log".
    capture_warnings : bool
        Whether to capture warnings in log. Default is True.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    logger = logging.getLogger(name)
    logging.basicConfig(
        level=level,
        filename=filename,
        filemode="w",
        format=FORMAT,
        encoding="utf-8",
    )
    logging.captureWarnings(capture=capture_warnings)

    return logger
