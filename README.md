# `janus-core`

![logo][logo]

[![PyPI version][pypi-badge]][pypi-link]
[![Python versions][python-badge]][python-link]
[![Build Status][ci-badge]][ci-link]
[![Coverage Status][cov-badge]][cov-link]
[![Docs status][docs-badge]][docs-link]
[![License][license-badge]][license-link]
[![DOI][doi-badge]][doi-link]

Tools for machine learnt interatomic potentials

## Getting started

### Dependencies

`janus-core` dependencies currently include:

- Python >= 3.9
- ASE >= 3.23
- mace-torch = 0.3.8
- chgnet = 0.3.8 (optional)
- matgl = 1.1.3 (optional)
- sevenn = 0.10.0 (optional)
- alignn = 2024.5.27 (optional)

All required and optional dependencies can be found in [pyproject.toml](pyproject.toml).

> [!NOTE]
> Where possible, we expect to update pinned MLIP dependencies to match their latest releases, subject to any required API fixes.


### Installation

The latest stable release of `janus-core`, including its dependencies, can be installed from PyPI by running:

```
python3 -m pip install janus-core
```

To get all the latest changes, `janus-core` can also be installed from GitHub:

```
python3 -m pip install git+https://github.com/stfc/janus-core.git
```

By default, MACE is the only MLIP installed.

Other MLIPs can be installed as `extras`. For example, to install CHGNet and M3GNet, run:

```python
python3 -m pip install janus-core[chgnet,m3gnet]
```

or to install all supported MLIPs:

```python
python3 -m pip install janus-core[all]
```

Individual `extras` are listed in [Getting Started](https://stfc.github.io/janus-core/getting_started/getting_started.html#installation), as well as in [pyproject.toml](pyproject.toml) under `[tool.poetry.extras]`.


### Further help

Please see [Getting Started](https://stfc.github.io/janus-core/getting_started/getting_started.html),
as well as guides for janus-core's [Python](https://stfc.github.io/janus-core/user_guide/python.html)
and [command line](https://stfc.github.io/janus-core/user_guide/command_line.html) interfaces,
for additional information, or [open an issue](https://github.com/stfc/janus-core/issues/new) if something doesn't seem right.


## Features

Unless stated otherwise, MLIP calculators and calculations rely heavily on [ASE](https://wiki.fysik.dtu.dk/ase/).

Current and planned features include:

- [x] Support for multiple MLIPs
  - MACE
  - M3GNet
  - CHGNet
  - ALIGNN (experimental)
  - SevenNet (experimental)
- [x] Single point calculations
- [x] Geometry optimisation
- [x] Molecular Dynamics
  - NVE
  - NVT (Langevin(Eijnden/Ciccotti flavour) and Nosé-Hoover (Melchionna flavour))
  - NPT (Nosé-Hoover (Melchiona flavour))
- [ ] Nudge Elastic Band
- [x] Phonons
  - Phonopy
- [x] Equation of State
- [x] Training ML potentials
  - MACE
- [x] Fine-tuning MLIPs
  - MACE
- [x] MLIP descriptors
  - MACE
- [x] Data preprocessing
  - MACE
- [ ] Rare events simulations
  - PLUMED

## Command line interface

All supported MLIP calculations are accessible through subcommands of the `janus` command line tool, which is installed with the package:

```shell
janus singlepoint
janus geomopt
janus md
janus phonons
janus eos
janus train
janus descriptors
janus preprocess
```

For example, a single point calcuation (using the [MACE-MP](https://github.com/ACEsuit/mace-mp) "small" force-field) can be performed by running:

```shell
janus singlepoint --struct tests/data/NaCl.cif --arch mace_mp --model-path small
```

A description of each subcommand, as well as valid options, can be listed using the `--help` option. For example,

```shell
janus singlepoint --help
```

prints the following:

```shell
Usage: janus singlepoint [OPTIONS]

  Perform single point calculations and save to file.

Options:
  --struct PATH        Path of structure to simulate.  [required]
  --arch TEXT          MLIP architecture to use for calculations.  [default:
                       mace_mp]
  --device TEXT        Device to run calculations on.  [default: cpu]
  --model-path TEXT    Path to MLIP model.  [default: None]
  --properties TEXT    Properties to calculate. If not specified, 'energy',
                       'forces' and 'stress' will be returned.
  --out PATH           Path to save structure with calculated results. Default
                       is inferred from name of structure file.
  --read-kwargs DICT   Keyword arguments to pass to ase.io.read. Must be
                       passed as a dictionary wrapped in quotes, e.g. "{'key'
                       : value}".  [default: "{}"]
  --calc-kwargs DICT   Keyword arguments to pass to selected calculator. Must
                       be passed as a dictionary wrapped in quotes, e.g.
                       "{'key' : value}". For the default architecture
                       ('mace_mp'), "{'model':'small'}" is set unless
                       overwritten.
  --write-kwargs DICT  Keyword arguments to pass to ase.io.write when saving
                       results. Must be passed as a dictionary wrapped in
                       quotes, e.g. "{'key' : value}".  [default: "{}"]
  --log PATH           Path to save logs to. Default is inferred from the name
                       of the structure file.
  --summary PATH       Path to save summary of inputs, start/end time, and
                       carbon emissions. Default is inferred from the name of
                       the structure file.
  --config TEXT        Configuration file.
  --help               Show this message and exit.
```

Please see the [user guide](https://stfc.github.io/janus-core/user_guide/command_line.html) for examples of each subcommand.


### Using configuration files

Default values for all command line options may be specifed through a Yaml 1.1 formatted configuration file by adding the `--config` option. If an option is present in both the command line and configuration file, the command line value takes precedence.

For example, with the following configuration file and command:

```yaml
struct: "NaCl.cif"
properties:
  - "energy"
out: "NaCl-results.extxyz"
arch: mace_mp
model-path: medium
calc-kwargs:
  dispersion: True
```

```shell
janus singlepoint --struct KCl.cif --out KCl-results.cif --config config.yml
```

This will run a singlepoint energy calculation on `KCl.cif` using the [MACE-MP](https://github.com/ACEsuit/mace-mp) "medium" force-field, saving the results to `KCl-results.cif`.

> [!NOTE]
> `properties` must be passed as a Yaml list, as above, not as a string.

Example configurations for all commands can be found in [janus-tutorials](https://github.com/stfc/janus-tutorials/tree/main/configs)


## Python interface

Calculations can also be run through the Python interface. For example, running:

```python
from janus_core.calculations.single_point import SinglePoint

single_point = SinglePoint(
    struct_path="tests/data/NaCl.cif",
    arch="mace_mp",
    model_path="tests/models/mace_mp_small.model",
)

results = single_point.run()
print(results)
```

will read the NaCl structure file and attach the MACE-MP (medium) calculator, before calculating and printing the energy, forces, and stress.

Jupyter Notebook tutorials illustrating the use of currently available calculations can be found in the [janus-tutorials](https://github.com/stfc/janus-tutorials) repository. This currently includes examples for:

- [Single Point](https://colab.research.google.com/github/stfc/janus-tutorials/blob/main/single_point.ipynb)
- [Geometry Optimization](https://colab.research.google.com/github/stfc/janus-tutorials/blob/main/geom_opt.ipynb)
- [Molecular Dynamics](https://colab.research.google.com/github/stfc/janus-tutorials/blob/main/md.ipynb)
- [Equation of State](https://colab.research.google.com/github/stfc/janus-tutorials/blob/main/eos.ipynb)
- [Phonons](https://colab.research.google.com/github/stfc/janus-tutorials/blob/main/phonons.ipynb)


## Calculation outputs

By default, calculations performed will modify the underlying [ase.Atoms](https://wiki.fysik.dtu.dk/ase/ase/atoms.html) object
to store information in the `Atoms.info` and `Atoms.arrays` dictionaries about the MLIP used.

Additional dictionary keys include `arch`, corresponding to the MLIP architecture used,
and `model_path`, corresponding to the model path, name or label.

Results from the MLIP calculator, which are typically stored in `Atoms.calc.results`, will also, by default,
be copied to these dictionaries, prefixed by the MLIP `arch`.

For example:

```python
from janus_core.calculations.single_point import SinglePoint

single_point = SinglePoint(
    struct_path="tests/data/NaCl.cif",
    arch="mace_mp",
    model_path="tests/models/mace_mp_small.model",
)

single_point.run()
print(single_point.struct.info)
```

will return

```python
{
  'spacegroup': Spacegroup(1, setting=1),
  'unit_cell': 'conventional',
  'occupancy': {'0': {'Na': 1.0}, '1': {'Cl': 1.0}, '2': {'Na': 1.0}, '3': {'Cl': 1.0}, '4': {'Na': 1.0}, '5': {'Cl': 1.0}, '6': {'Na': 1.0}, '7': {'Cl': 1.0}},
  'model_path': 'tests/models/mace_mp_small.model',
  'arch': 'mace_mp',
  'mace_mp_energy': -27.035127799332745,
  'mace_mp_stress': array([-4.78327600e-03, -4.78327600e-03, -4.78327600e-03,  1.08000967e-19, -2.74004242e-19, -2.04504710e-19]),
  'system_name': 'NaCl',
}
```

> [!NOTE]
> If running calculations with multiple MLIPs, `arch` and `mlip_model` will be overwritten with the most recent MLIP information.
> Results labelled by the architecture (e.g. `mace_mp_energy`) will be saved between MLIPs,
> unless the same `arch` is chosen, in which case these values will also be overwritten.

This is also the case the calculations performed using the CLI, with the same information written to extxyz output files.

> [!TIP]
> For complete provenance tracking, calculations and training can be run using the [aiida-mlip](https://github.com/stfc/aiida-mlip/) AiiDA plugin.


## Development

We recommend installing poetry for dependency management when developing for `janus-core`:

1. Install [poetry](https://python-poetry.org/docs/#installation)
2. (Optional) Create a virtual environment
3. Install `janus-core` with dependencies:

```shell
git clone https://github.com/stfc/janus-core
cd janus-core
python3 -m pip install --upgrade pip
poetry install --with pre-commit,dev,docs  # install with useful dev dependencies
pre-commit install  # install pre-commit hooks
pytest -v  # discover and run all tests
```


## License

[BSD 3-Clause License](LICENSE)


## Funding

Contributors to this project were funded by

[![PSDI](https://raw.githubusercontent.com/stfc/janus-core/main/docs/source/images/psdi-100.webp)](https://www.psdi.ac.uk/)
[![ALC](https://raw.githubusercontent.com/stfc/janus-core/main/docs/source/images/alc-100.webp)](https://adalovelacecentre.ac.uk/)
[![CoSeC](https://raw.githubusercontent.com/stfc/janus-core/main/docs/source/images/cosec-100.webp)](https://www.scd.stfc.ac.uk/Pages/CoSeC.aspx)


[ci-badge]: https://github.com/stfc/janus-core/actions/workflows/ci.yml/badge.svg?branch=main
[ci-link]: https://github.com/stfc/janus-core/actions
[cov-badge]: https://coveralls.io/repos/github/stfc/janus-core/badge.svg?branch=main
[cov-link]: https://coveralls.io/github/stfc/janus-core?branch=main
[docs-badge]: https://github.com/stfc/janus-core/actions/workflows/docs.yml/badge.svg
[docs-link]: https://stfc.github.io/janus-core/
[pypi-badge]: https://badge.fury.io/py/janus-core.svg
[pypi-link]: https://pypi.org/project/janus-core/
[python-badge]: https://img.shields.io/pypi/pyversions/janus-core.svg
[python-link]: https://pypi.org/project/janus-core/
[license-badge]: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
[license-link]: https://opensource.org/licenses/BSD-3-Clause
[doi-link]: https://zenodo.org/badge/latestdoi/754081470
[doi-badge]: https://zenodo.org/badge/754081470.svg
[logo]: https://raw.githubusercontent.com/stfc/janus-core/main/docs/source/images/janus-core-100.png
