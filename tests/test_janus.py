"""Test importing janus_core."""

from __future__ import annotations

import janus_core


def test_import():
    """Test janus_core has been imported successfully."""
    assert janus_core.__version__ is not None
