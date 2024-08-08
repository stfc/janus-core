"""Test FileNameMixin functions."""

from pathlib import Path

from ase.io import read
import pytest

from janus_core.helpers.utils import FileNameMixin

DATA_PATH = Path(__file__).parent / "data"
STRUCT = read(DATA_PATH / "benzene.xyz")


class DummyFileHandler(FileNameMixin):  # pylint: disable=too-few-public-methods
    """Used for testing FileNameMixin methods."""

    def build_filename(self, *args, **kwargs):
        """
        Expose _build_filename publicly.
        """
        return self._build_filename(*args, **kwargs)


@pytest.mark.parametrize(
    "params,file_prefix",
    (
        # Defaults to structure atoms from ASE
        ((STRUCT, None), "C6H6"),
        # file_prefix just sets itself
        ((STRUCT, "benzene"), "benzene"),
        # file_prefix ignores additional
        ((STRUCT, "benzene", "wowzers"), "benzene"),
        # Additional only applies where no file_prefix
        ((STRUCT, None, "wowzers"), "C6H6-wowzers"),
    ),
)
def test_file_name_mixin_init(params, file_prefix):
    """Test various options for initializing the mixin."""
    file_mix = DummyFileHandler(*params)

    assert file_mix.file_prefix == Path(file_prefix)


@pytest.mark.parametrize(
    "mixin_params,file_args,file_kwargs,file_name",
    (
        ((STRUCT, None), ("data.xyz",), {}, "C6H6-data.xyz"),
        ((STRUCT, "benzene"), ("data.xyz",), {}, "benzene-data.xyz"),
        ((STRUCT, "benzene", "wowzers"), ("data.xyz",), {}, "benzene-data.xyz"),
        ((STRUCT, None, "wowzers"), ("data.xyz",), {}, "C6H6-wowzers-data.xyz"),
        # Additional stacks with base
        (
            (STRUCT, None, "wowzers"),
            ("data.xyz", "beef"),
            {},
            "C6H6-wowzers-beef-data.xyz",
        ),
        # Prefix override ignores class options
        (
            (STRUCT, None, "wowzers"),
            ("data.xyz",),
            {"prefix_override": "beef"},
            "beef-data.xyz",
        ),
        # But not additional
        (
            (STRUCT, None, "wowzers"),
            ("data.xyz", "tasty"),
            {"prefix_override": "beef"},
            "beef-tasty-data.xyz",
        ),
        # Filename overrides everything
        (
            (STRUCT, None, "wowzers"),
            ("data.xyz",),
            {"filename": "hello.xyz"},
            "hello.xyz",
        ),
        (
            (STRUCT, None, "wowzers"),
            ("data.xyz",),
            {"prefix_override": "beef", "filename": "hello.xyz"},
            "hello.xyz",
        ),
        (
            (STRUCT, None, "wowzers"),
            ("data.xyz", "tasty"),
            {"prefix_override": "beef", "filename": "hello.xyz"},
            "hello.xyz",
        ),
    ),
)
def test_file_name_mixin_build(mixin_params, file_args, file_kwargs, file_name):
    """Test building the filename for mixins."""
    file_mix = DummyFileHandler(*mixin_params)

    assert file_mix.build_filename(*file_args, **file_kwargs) == Path(file_name)
