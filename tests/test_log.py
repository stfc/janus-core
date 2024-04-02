"""Test logging module."""

import yaml

from janus_core.log import config_logger


def test_multiline_log(tmp_path):
    """Test multiline log is written correctly."""
    log_path = tmp_path / "test.log"

    logger = config_logger(name=__name__, filename=log_path, force=True)
    logger.info(
        """
        Line 1
        Line 2
        Line 3
        """
    )
    assert log_path.exists()
    with open(log_path, encoding="utf8") as log:
        log_dicts = yaml.safe_load(log)

    assert log_dicts[0]["level"] == "INFO"
    assert len(log_dicts[0]["message"]) == 3
