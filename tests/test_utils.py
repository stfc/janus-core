"""Test utility functions."""

from pathlib import Path

from janus_core.helpers.utils import dict_paths_to_strs, dict_remove_hyphens


def test_dict_paths_to_strs():
    """Test Paths are converted to strings."""
    dictionary = {
        "key1": Path("/example/path"),
        "key2": {
            "key3": Path("another/example"),
            "key4": "example",
        },
    }

    # Check Paths are present
    assert isinstance(dictionary["key1"], Path)
    assert isinstance(dictionary["key2"]["key3"], Path)
    assert not isinstance(dictionary["key1"], str)

    dict_paths_to_strs(dictionary)

    # Check Paths are now strings
    assert isinstance(dictionary["key1"], str)
    assert isinstance(dictionary["key2"]["key3"], str)


def test_dict_remove_hyphens():
    """Test hyphens are replaced with underscores."""
    dictionary = {
        "key-1": "value_1",
        "key-2": {
            "key-3": "value-3",
            "key-4": 4,
            "key_5": 5.0,
            "key6": {"key-7": "value7"},
        },
    }
    dictionary = dict_remove_hyphens(dictionary)

    # Check hyphens are now strings
    assert dictionary["key_1"] == "value_1"
    assert dictionary["key_2"]["key_3"] == "value-3"
    assert dictionary["key_2"]["key_4"] == 4
    assert dictionary["key_2"]["key_5"] == 5.0
    assert dictionary["key_2"]["key6"]["key_7"] == "value7"
