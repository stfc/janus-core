"""Test logging module."""

from __future__ import annotations

import yaml

from janus_core.helpers.log import config_logger
from tests.utils import assert_log_contains


def test_multiline_log(tmp_path):
    """Test multiline log is written correctly."""
    log_path = tmp_path / "test.log"

    logger = config_logger(name=__name__, filename=log_path)
    logger.info(
        """
        Line 1
        Line 2
        Line 3
        """
    )
    logger.info("Line 4")

    assert log_path.exists()
    with open(log_path, encoding="utf8") as log:
        log_dicts = yaml.safe_load(log)

    assert log_dicts[0]["level"] == "INFO"
    assert len(log_dicts[0]["message"]) == 3

    assert_log_contains(log_path, includes=["Line 1", "Line 4"])
