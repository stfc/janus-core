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
"""


class CustomFormatter(logging.Formatter):
    """
    Custom formatter to convert multiline messages into yaml list.

    Methods
    -------
    format(record)
        Format log message to convert new lines into a yaml list.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log message to convert new lines into a yaml list.

        Parameters
        ----------
        record : logging.LogRecord
            Log record to be formatted.

        Returns
        -------
        str
            Formatted log message.
        """
        # Replace "" with '' to prevent invalid wrapping
        record.msg = record.msg.replace('"', "'")

        # Convert new lines into yaml list
        if len(record.msg.split("\n")) > 1:
            msg = record.msg
            record.msg = "\n"
            for line in msg.split("\n"):
                # Exclude empty lines
                if line.strip():
                    record.msg += f'    - "{line.strip()}"\n'
            # Remove final newline (single character)
            record.msg = record.msg[:-1]
        else:
            # Wrap line in quotes here rather than in FORMAT due to list
            record.msg = f'"{record.msg}"'

        return super().format(record)


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

        handler = logging.FileHandler(
            filename,
            filemode,
            encoding="utf-8",
        )
        handler.setLevel(level)
        formatter = CustomFormatter(FORMAT)
        handler.setFormatter(formatter)

        logging.basicConfig(
            level=level,
            force=force,
            handlers=[handler],
        )
        logging.captureWarnings(capture=capture_warnings)
    else:
        logger = None
    return logger
