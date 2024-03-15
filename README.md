[![Build Status][ci-badge]][ci-link]
[![Coverage Status][cov-badge]][cov-link]
[![Docs status][docs-badge]][docs-link]
[![PyPI version][pypi-badge]][pypi-link]
[![License][license-badge]][license-link]
[![DOI][doi-badge]][doi-link]

# janus-core

Tools for machine learnt interatomic potentials

## Features in development

- [x] Support for multiple MLIPs
  - MACE
  - M3GNET
  - CHGNET
- [x] Single point calculations
- [ ] Geometry optimisation
- [ ] Molecular Dynamics
  - NVE
  - NVT (Langevin(Eijnden/Ciccotti flavour) and Nosé-Hoover (Melchionna flavour))
  - NPT (Nosé-Hoover (Melchiona flavour))
- [ ] Nudge Elastic Band
- [ ] Phonons
  - vibroscopy
- [ ] Training ML potentials
  - MACE
- [ ] Fine tunning MLIPs
  - MACE
- [ ] Rare events simulations
  - PLUMED

The code relies heavily on ASE, unless something else is mentioned.

## Development

1. Install [poetry](https://python-poetry.org/docs/#installation)
2. (Optional) Create a virtual environment
3. Install `janus-core` with dependencies:

```shell
git clone https://github.com/stfc/janus-core
cd janus-core
pip install --upgrade pip
poetry install --with pre-commit,dev,docs  # install extra dependencies
pre-commit install  # install pre-commit hooks
pytest -v  # discover and run all tests
```

Manually updating ASE via https://gitlab.com/ase/ase is strongly recommended, as tags are no longer regularly published. For example:

```shell
pip install git+https://gitlab.com/ase/ase.git@b31569210d739bd12c8ad2b6ec0290108e049eea
```

To prevent poetry downgrading ASE when installing in future, add the commit to pyproject.toml:

```shell
poetry add git+https://gitlab.com:ase/ase.git#b31569210d739bd12c8ad2b6ec0290108e049eea
```

## Examples

Perform a single point calcuation:
```shell
janus singlepoint --struct tests/data/NaCl.cif
```

This will calculate the energy, stress and forces and save this in `NaCl-results.xyz`, in addition to generating a log file, `singlepoint.log`.

Additional options may be specified. For example:

```shell
janus singlepoint --struct tests/data/NaCl.cif --property energy --arch mace_mp --calc-kwargs "{'model' : './tests/models/mace_mp_small.model'}" --log './example.log' --write-kwargs "{'filename': './example.xyz'}"
```

This defines the MLIP architecture and path to the saved model, as well as changing where the log and results files are saved.

Note: the MACE calculator currently returns energy, forces and stress together, so in this case the choice of property will not change the output.


Perform geometry optimization:
```shell
janus geomopt --struct tests/data/H2O.cif
```

This will calculate optimize the atomic positions and save the resulting structure in `H2O-opt.xyz`, in addition to generating a log file, `geomopt.log`.

Additional options may be specified. This shares most options with `singlepoint`, as well as a few additional options, such as:

```shell
janus geomopt --struct tests/data/NaCl.cif --fully-opt --vectors-only --traj 'NaCl-traj.xyz'
```

This allows the cell to be optimised, allowing only hydrostatic deformation, and saves the optimization trajector in addition to the final structure and log.

## License

[BSD 3-Clause License](LICENSE)

## Funding

Contributors to this project were funded by

[![PSDI](https://raw.githubusercontent.com/stfc/janus-core/main/docs/source/images/psdi-100.webp)](https://www.psdi.ac.uk/)
[![ALC](https://raw.githubusercontent.com/stfc/janus-core/main/docs/source/images/alc-100.webp)](https://adalovelacecentre.ac.uk/)
[![CoSeC](https://raw.githubusercontent.com/stfc/janus-core/main/docs/source/images/cosec-100.webp)](https://www.scd.stfc.ac.uk/Pages/CoSeC.aspx)


[ci-badge]: https://github.com/stfc/janus-core/workflows/ci/badge.svg
[ci-link]: https://github.com/stfc/janus-core/actions
[cov-badge]: https://coveralls.io/repos/github/stfc/janus-core/badge.svg?branch=main
[cov-link]: https://coveralls.io/github/stfc/janus-core?branch=main
[docs-badge]: https://github.com/stfc/janus-core/actions/workflows/docs.yml/badge.svg
[docs-link]: https://stfc.github.io/janus-core/
[pypi-badge]: https://badge.fury.io/py/janus-core.svg
[pypi-link]: https://badge.fury.io/py/janus-core
[license-badge]: https://img.shields.io/badge/License-BSD_3--Clause-blue.svg
[license-link]: https://opensource.org/licenses/BSD-3-Clause
[doi-link]: https://zenodo.org/badge/latestdoi/754081470
[doi-badge]: https://zenodo.org/badge/754081470.svg
