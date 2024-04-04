"""Configure logger with yaml-styled format."""

import logging
from typing import Literal, Optional

from janus_core.janus_types import LogLevel


class YamlFormatter(logging.Formatter):  # numpydoc ignore=PR02
    """
    Custom formatter to convert multiline messages into yaml list.

    Parameters
    ----------
    fmt : Optional[str]
        A format string in the given style for the logged output as a whole. Default is
        defined by `FORMAT`.
    datefmt : Optional[str]
        A format string in the given style for the date/time portion of the logged
        output. Default is taken from logging.Formatter.formatTime().
    style : Literal['%', '{', '$']
        Determines how the format string will be merged with its data. Can be one of
        '%', '{' or '$'. Default is '%'.
    validate : bool
        If True, incorrect or mismatched fmt and style will raise a ValueError. Default
        is True.
    defaults : Optional[dict[str, logging.Any]]
         A dictionary with default values to use in custom fields. Default is None.
    *args
        Arguments to pass to logging.Formatter.__init__.
    **kwargs
        Keyword arguments to pass to logging.Formatter.__init__.

    Methods
    -------
    format(record)
        Format log message to convert new lines into a yaml list.
    """

    FORMAT = """
- timestamp: %(asctime)s
  level: %(levelname)s
  message: %(message)s
  trace: %(module)s
  line: %(lineno)d
"""

    def __init__(self, *args, **kwargs) -> None:
        """
        Set default string format to yaml style.

        Parameters
        ----------
        *args
            Arguments to pass to logging.Formatter.__init__.
        **kwargs
            Keyword arguments to pass to logging.Formatter.__init__.
        """
        kwargs.setdefault("fmt", self.FORMAT)
        super().__init__(*args, **kwargs)

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
        record.msg = "\n" + "\n".join(
            f'    - "{line.strip()}"'
            for line in record.msg.splitlines()
            if line.strip()
        )
        return super().format(record)


def config_logger(
    name: str,
    filename: Optional[str] = None,
    level: LogLevel = logging.INFO,
    capture_warnings: bool = True,
    filemode: Literal["r", "w", "a", "x", "r+", "w+", "a+", "x+"] = "w",
    force: bool = True,
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
        configuration. Default is True.

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
        formatter = YamlFormatter()
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
