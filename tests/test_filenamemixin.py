"""Test FileNameMixin functions."""

from __future__ import annotations

from pathlib import Path

from ase.io import read
import pytest

from janus_core.helpers.utils import FileNameMixin

DATA_PATH = Path(__file__).parent / "data"
STRUCT = read(DATA_PATH / "benzene.xyz")


class DummyFileHandler(FileNameMixin):
    """Used for testing FileNameMixin methods."""

    def build_filename(self, *args, **kwargs):
        """Expose _build_filename publicly."""
        return self._build_filename(*args, **kwargs)


@pytest.mark.parametrize(
    "params,file_prefix",
    (
        # Defaults to structure atoms from ASE
        ((STRUCT, None, None), "C6H6"),
        # Defaults to stem over ASE structure
        ((STRUCT, "mybenzene", None), "mybenzene"),
        # file_prefix just sets itself
        ((STRUCT, "mybenzene", "benzene"), "benzene"),
        # file_prefix ignores additional
        ((STRUCT, None, "benzene", "wowzers"), "benzene"),
        # Additional only applies where no file_prefix
        ((STRUCT, None, None, "wowzers"), "C6H6-wowzers"),
        # Additional only applies where no file_prefix
        ((STRUCT, "mybenzene", None, "wowzers"), "mybenzene-wowzers"),
    ),
)
def test_file_name_mixin_init(params, file_prefix):
    """Test various options for initializing the mixin."""
    file_mix = DummyFileHandler(*params)

    assert file_mix.file_prefix == Path(file_prefix)


@pytest.mark.parametrize(
    "mixin_params,file_args,file_kwargs,file_name",
    (
        ((STRUCT, None, None), ("data.xyz",), {}, "C6H6-data.xyz"),
        ((STRUCT, None, "benzene"), ("data.xyz",), {}, "benzene-data.xyz"),
        ((STRUCT, None, "benzene", "wowzers"), ("data.xyz",), {}, "benzene-data.xyz"),
        ((STRUCT, None, None, "wowzers"), ("data.xyz",), {}, "C6H6-wowzers-data.xyz"),
        # Additional stacks with base
        (
            (STRUCT, None, None, "wowzers"),
            ("data.xyz", "beef"),
            {},
            "C6H6-wowzers-beef-data.xyz",
        ),
        (
            (STRUCT, "mybenzene", None, "wowzers"),
            ("data.xyz", "beef"),
            {},
            "mybenzene-wowzers-beef-data.xyz",
        ),
        # but not file_prefix
        (
            (STRUCT, "mybenzene", "benzene", "wowzers"),
            ("data.xyz", "beef"),
            {},
            "benzene-beef-data.xyz",
        ),
        # Prefix override ignores class options
        (
            (STRUCT, None, None, "wowzers"),
            ("data.xyz",),
            {"prefix_override": "beef"},
            "beef-data.xyz",
        ),
        # But not additional
        (
            (STRUCT, None, None, "wowzers"),
            ("data.xyz", "tasty"),
            {"prefix_override": "beef"},
            "beef-tasty-data.xyz",
        ),
        # Filename overrides everything
        (
            (STRUCT, None, None, "wowzers"),
            ("data.xyz",),
            {"filename": "hello.xyz"},
            "hello.xyz",
        ),
        (
            (STRUCT, None, None, "wowzers"),
            ("data.xyz",),
            {"prefix_override": "beef", "filename": "hello.xyz"},
            "hello.xyz",
        ),
        (
            (STRUCT, None, None, "wowzers"),
            ("data.xyz", "tasty"),
            {"prefix_override": "beef", "filename": "hello.xyz"},
            "hello.xyz",
        ),
        (
            (STRUCT, "mybenzene", "benzene", "wowzers"),
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
