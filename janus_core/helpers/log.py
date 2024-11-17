"""Configure logger with yaml-styled format."""

from __future__ import annotations

import json
import logging
from typing import Literal

from codecarbon import OfflineEmissionsTracker
from codecarbon.output import LoggerOutput

from janus_core.helpers.janus_types import LogLevel


class YamlFormatter(logging.Formatter):  # numpydoc ignore=PR02
    """
    Custom formatter to convert multiline messages into yaml list.

    Parameters
    ----------
    fmt : str | None
        A format string in the given style for the logged output as a whole. Default is
        defined by `FORMAT`.
    datefmt : str | None
        A format string in the given style for the date/time portion of the logged
        output. Default is taken from logging.Formatter.formatTime().
    style : Literal['%', '{', '$']
        Determines how the format string will be merged with its data. Can be one of
        '%', '{' or '$'. Default is '%'.
    validate : bool
        If True, incorrect or mismatched fmt and style will raise a ValueError. Default
        is True.
    defaults : dict[str, logging.Any] | None
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
        # Parse JSON dump from codecarbon
        record.msg = str(record.msg)
        if len(record.msg) > 1 and (record.msg[0] == "{" and record.msg[-1] == "}"):
            msg_dict = json.loads(record.msg)
            record.msg = ""
            for key, value in msg_dict.items():
                record.msg += f"\n    {key}: {value}"
        else:
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
    filename: str | None = None,
    level: LogLevel = logging.INFO,
    capture_warnings: bool = True,
    filemode: Literal["r", "w", "a", "x", "r+", "w+", "a+", "x+"] = "w",
    force: bool = True,
) -> logging.Logger | None:
    """
    Configure logger with yaml-styled format.

    Parameters
    ----------
    name : str
        Name of logger. Default is None.
    filename : str | None
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
    logging.Logger | None
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


def config_tracker(
    janus_logger: logging.Logger | None,
    track_carbon: bool = True,
    *,
    country_iso_code: str = "GBR",
    save_to_file: bool = False,
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "critical",
) -> OfflineEmissionsTracker | None:
    """
    Configure codecarbon tracker to log outputs.

    Parameters
    ----------
    janus_logger : logging.Logger | None
        Logger for codecarbon output.
    track_carbon : bool
        Whether to track carbon emissions for calculation. Default is True.
    country_iso_code : str
        3 letter ISO Code of the country where the code is running. Default is "GBR".
    save_to_file : bool
        Whether to also output results to a csv file. Default is False.
    log_level : Literal["debug", "info", "warning", "error", "critical"]
        Log level of internal carbon tracker log. Default is "critical".

    Returns
    -------
    OfflineEmissionsTracker | None
        Configured offline codecarbon tracker, if logger is specified.
    """
    if janus_logger and track_carbon:
        carbon_logger = LoggerOutput(janus_logger)
        tracker = OfflineEmissionsTracker(
            country_iso_code=country_iso_code,
            save_to_file=save_to_file,
            save_to_logger=True,
            logging_logger=carbon_logger,
            project_name="janus-core",
            log_level=log_level,
            allow_multiple_runs=True,
        )

        # Suppress further logging from codecarbon
        carbon_logger = logging.getLogger("codecarbon")
        while carbon_logger.hasHandlers():
            carbon_logger.removeHandler(carbon_logger.handlers[0])

    else:
        tracker = None

    return tracker
