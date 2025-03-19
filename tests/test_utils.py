"""Test utility functions."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import read
import pytest

from janus_core.cli.types import TyperDict
from janus_core.cli.utils import (
    dict_paths_to_strs,
    dict_remove_hyphens,
    parse_correlation_kwargs,
)
from janus_core.helpers.janus_types import SliceLike, StartStopStep
from janus_core.helpers.mlip_calculators import choose_calculator
from janus_core.helpers.struct_io import output_structs
from janus_core.helpers.utils import (
    none_to_dict,
    selector_len,
    slicelike_to_startstopstep,
    validate_slicelike,
)
from janus_core.processing.observables import Stress, Velocity

DATA_PATH = Path(__file__).parent / "data/NaCl.cif"
MODEL_PATH = Path(__file__).parent / "models/mace_mp_small.model"


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


@pytest.mark.parametrize("arch", ["mace_mp", "chgnet", "sevennet"])
@pytest.mark.parametrize("write_results", [True, False])
@pytest.mark.parametrize("properties", [None, ["energy"], ["energy", "forces"]])
@pytest.mark.parametrize("invalidate_calc", [True, False])
@pytest.mark.parametrize(
    "write_kwargs", [{}, {"write_results": False}, {"set_info": False}]
)
def test_output_structs(
    arch, write_results, properties, invalidate_calc, write_kwargs, tmp_path
):
    """Test output_structs copies/moves results to Atoms.info and writes files."""
    if arch == "chgnet":
        pytest.importorskip("chgnet")
    if arch == "sevennet":
        pytest.importorskip("sevenn")

    struct = read(DATA_PATH)
    struct.calc = choose_calculator(arch=arch)

    if properties:
        results_keys = set(properties)
    else:
        results_keys = {"energy", "forces", "stress"}

    if arch == "mace_mp":
        model = "small"
    if arch == "chgnet":
        model = "0.4.0"
    if arch == "sevennet":
        model = "SevenNet-0_11July2024"

    label_keys = {f"{arch}_{key}" for key in results_keys}

    write_kwargs = {}
    output_file = tmp_path / "output.extxyz"
    if write_results:
        write_kwargs["filename"] = output_file

    # Use calculator
    struct.get_potential_energy()
    struct.get_stress()

    # Check all expected keys are in results
    assert results_keys <= struct.calc.results.keys()

    # Check results and MLIP-labelled keys are not in info or arrays
    assert not results_keys & struct.info.keys()
    assert not results_keys & struct.arrays.keys()
    assert not label_keys & struct.info.keys()
    assert not label_keys & struct.arrays.keys()

    output_structs(
        struct,
        write_results=write_results,
        properties=properties,
        invalidate_calc=invalidate_calc,
        write_kwargs=write_kwargs,
    )

    # Check results keys depend on invalidate_calc
    if invalidate_calc:
        assert not results_keys & struct.calc.results.keys()
    else:
        assert results_keys <= struct.calc.results.keys()

    # Check labelled keys added to info and arrays
    if "set_info" not in write_kwargs or write_kwargs["set_info"]:
        assert label_keys <= struct.info.keys() | struct.arrays.keys()
        assert struct.info["arch"] == arch
        assert struct.info["model_path"] == model

    # Check file written correctly if write_results
    if write_results:
        assert output_file.exists()
        atoms = read(output_file)
        assert isinstance(atoms, Atoms)

        # Check labelled info and arrays was written and can be read back in
        if "set_info" not in write_kwargs or write_kwargs["set_info"]:
            assert label_keys <= atoms.info.keys() | atoms.arrays.keys()
            assert atoms.info["arch"] == arch
            assert atoms.info["model_path"] == model

        # Check calculator results depend on invalidate_calc
        if invalidate_calc:
            assert atoms.calc is None
        elif "write_results" not in write_kwargs or write_kwargs["write_results"]:
            assert results_keys <= atoms.calc.results.keys()

    else:
        assert not output_file.exists()


@pytest.mark.parametrize(
    "dicts_in",
    [
        [None, {"a": 1}, {}, {"b": 2, "a": 11}, None],
        (None, {"a": 1}, {}, {"b": 2, "a": 11}, None),
    ],
)
def test_none_to_dict(dicts_in):
    """Test none_to_dict removes Nones from sequence, and preserves dictionaries."""
    dicts = list(none_to_dict(*dicts_in))
    for dictionary in dicts:
        assert dictionary is not None

    assert dicts[0] == {}
    assert dicts[1] == dicts_in[1]
    assert dicts[2] == dicts_in[2]
    assert dicts[3] == dicts_in[3]
    assert dicts[4] == {}


@pytest.mark.parametrize(
    "slc, expected",
    [
        ((1, 2, 3), (1, 2, 3)),
        (1, (1, 2, 1)),
        (range(1, 2, 3), (1, 2, 3)),
        (slice(1, 2, 3), (1, 2, 3)),
        (-1, (-1, None, 1)),
        (range(10), (0, 10, 1)),
        (slice(0, None, 1), (0, None, 1)),
    ],
)
def test_slicelike_to_startstopstep(slc: SliceLike, expected: StartStopStep):
    """Test converting SliceLike to StartStopStep."""
    assert slicelike_to_startstopstep(slc) == expected


@pytest.mark.parametrize(
    "slc, len, expected",
    [
        ((1, 2, 3), 3, 1),
        (1, 1, 1),
        (range(1, 2, 3), 3, 1),
        (slice(1, 2, 3), 3, 1),
        (-1, 5, 1),
        (-3, 4, 1),
        (range(10), 10, 10),
        (slice(0, None, 2), 10, 5),
        ([-2, -1, 0], 9, 3),
        ([-1], 10, 1),
        ([0, -1, 2, 9], 10, 4),
    ],
)
def test_selector_len(slc: SliceLike | list[int], len: int, expected: int):
    """Test converting SliceLike to StartStopStep."""
    assert selector_len(slc, len) == expected


@pytest.mark.parametrize(
    "slc",
    [
        slice(0, 1, 1),
        slice(0, None, 1),
        range(3),
        range(0, 10, 1),
        -1,
        0,
        1,
        (0),
        (0, 1, 1),
        (0, None, 1),
    ],
)
def test_valid_slicelikes(slc):
    """Test validate_slicelike on valid SliceLikes."""
    validate_slicelike(slc)


@pytest.mark.parametrize(
    "slc",
    [
        1.0,
        "",
        None,
        [1],
        (None, 0, None),
        (0, 1, None),
        (None, None, None),
        (0, 1),
        (0, 1, 2, 3),
    ],
)
def test_invalid_slicelikes(slc):
    """Test validate_slicelike on invalid SliceLikes."""
    with pytest.raises(ValueError):
        validate_slicelike(slc)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"vaf": {"a": "Velocity"}}, "Correlation keyword argument points"),
        ({"vaf": {"points": 10}}, "At least one observable"),
        (
            {"vaf": {"points": 10, "a": "Velocity", "a_kwargs": ".", "b_kwargs": "."}},
            "cannot 'ditto' each other",
        ),
    ],
)
def test_parse_correlation_kwargs_errors(kwargs, message):
    """Test parsing of correlation CLI kwargs with ValueErrors."""
    with pytest.raises(ValueError, match=message):
        parse_correlation_kwargs(TyperDict(kwargs))


def test_parse_correlation_kwargs_unsupported_observable():
    """Test parsing of correlation CLI kwargs with a unsupported Observable."""
    with pytest.raises(
        AttributeError, match="'janus_core.processing.observables' has no attribute"
    ):
        parse_correlation_kwargs(
            TyperDict({"noa": {"a": "NOT_AN_OBSERVABLE", "points": 10}})
        )


def test_parse_correlation_kwargs():
    """Test parsing correlation CLI kwargs."""
    kwargs = {
        "vaf": {"a": "Velocity", "points": 100, "b_kwargs": {"components": ["x"]}},
        "vaf_ditto": {
            "b": "Velocity",
            "points": 314,
            "a_kwargs": {"atoms_slice": slice(0, None, 2)},
            "b_kwargs": ".",
        },
        "saf": {"a": "Stress", "points": 10, "a_kwargs": {"components": ["xy"]}},
        "vcross": {
            "a": "Velocity",
            "b": "Velocity",
            "points": 10,
            "a_kwargs": {"components": ["x"]},
        },
        "vcross_ditto": {
            "a": "Velocity",
            "b": "Velocity",
            "points": 10,
            "a_kwargs": {"components": ["x"]},
            "b_kwargs": ".",
        },
    }
    parsed = parse_correlation_kwargs(TyperDict(kwargs))
    assert len(parsed) == 5

    assert parsed[0]["name"] == "vaf"
    # b copies a.
    assert type(parsed[0]["a"]) is type(parsed[0]["b"]) is Velocity
    # a does not copy b's kwargs.
    assert parsed[0]["a"].components == ["x", "y", "z"]
    assert parsed[0]["b"].components == ["x"]
    assert parsed[0]["points"] == 100

    assert parsed[1]["name"] == "vaf_ditto"
    # a copies b.
    assert type(parsed[1]["a"]) is type(parsed[1]["b"]) is Velocity
    assert parsed[1]["a"].components == parsed[1]["b"].components == ["x", "y", "z"]
    # b copies a's kwargs
    assert parsed[1]["a"].atoms_slice == parsed[1]["b"].atoms_slice == slice(0, None, 2)
    assert parsed[1]["points"] == 314

    assert parsed[2]["name"] == "saf"
    # b copies a.
    assert type(parsed[2]["a"]) is type(parsed[2]["b"]) is Stress
    assert parsed[2]["points"] == 10
    # b copies a's kwargs (same Observable type).
    assert parsed[2]["a"].components == parsed[2]["b"].components == ["xy"]

    assert parsed[3]["name"] == "vcross"
    assert type(parsed[3]["a"]) is type(parsed[3]["b"]) is Velocity
    assert parsed[3]["points"] == 10
    assert parsed[3]["a"].components == ["x"]
    # b's kwargs were not copied from a.
    assert parsed[3]["b"].components == ["x", "y", "z"]

    assert parsed[4]["name"] == "vcross_ditto"
    assert type(parsed[4]["a"]) is type(parsed[4]["b"]) is Velocity
    assert parsed[4]["points"] == 10
    assert parsed[4]["a"].components == parsed[4]["b"].components == ["x"]
