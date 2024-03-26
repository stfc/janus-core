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
- [x] Geometry optimisation
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

### Single Point Calculations

Perform a single point calcuation (using the [MACE-MP](https://github.com/ACEsuit/mace-mp) "small" force-field):
```shell
janus singlepoint --struct tests/data/NaCl.cif --arch mace_mp --calc-kwargs "{'model' : 'small'}"
```

This will calculate the energy, stress and forces and save this in `NaCl-results.xyz`, in addition to generating a log file, `singlepoint.log`.

Additional options may be specified. For example:

```shell
janus singlepoint --struct tests/data/NaCl.cif --arch mace --calc-kwargs "{'model' : '/path/to/your/ml.model'}" --property energy --log './example.log' --write-kwargs "{'filename': './example.xyz'}"
```

This defines the MLIP architecture and path to your locally saved model, as well as changing where the log and results files are saved.

Note: the MACE calculator currently returns energy, forces and stress together, so in this case the choice of property will not change the output.

For all options, run `janus singlepoint --help`.


### Geometry optimization

Perform geometry optimization (using the [MACE-MP](https://github.com/ACEsuit/mace-mp) "small" force-field):

```shell
janus geomopt --struct tests/data/H2O.cif --arch mace_mp --calc-kwargs "{'model' : 'small'}"
```

This will optimize the atomic positions and save the resulting structure in `H2O-opt.xyz`, in addition to generating a log file, `geomopt.log`.

Additional options may be specified. This shares most options with `singlepoint`, as well as a few additional options, such as:

```shell
janus geomopt --struct tests/data/NaCl.cif --arch mace_mp --calc-kwargs "{'model' : 'small'}" --vectors-only --traj 'NaCl-traj.xyz'
```

This allows the cell to be optimised, allowing only hydrostatic deformation, and saves the optimization trajector in addition to the final structure and log.

For all options, run `janus geomopt --help`.


### Molecular dynamics

Run an NPT molecular dynamics simulation (using the [MACE-MP](https://github.com/ACEsuit/mace-mp) "small" force-field) at 300K and 1 bar for 1000 steps (1 ps):

```shell
janus md --ensemble npt --struct tests/data/NaCl.cif --arch mace_mp --calc-kwargs "{'model' : 'small'}" --temp 300 --steps 1000 --pressure 1.0
```

This will generate several output files:

- Thermodynamical statistics every 100 steps, written to `NaCl-npt-T300.0-p1.0-stats.dat`
- The structure trajectory every 100 steps, written to `NaCl-npt-T300.0-p1.0-traj.xyz`
- The structure to be able to restart the dynamics every 1000 steps, written to `NaCl-npt-T300.0-p1.0-res-1000.xyz`
- A log of the processes carried out, written to `md.log`


Additional options may be specified. For example:

```shell
janus md --ensemble nvt --struct tests/data/NaCl.cif --steps 1000 --timestep 0.5 --temp 300 --minimize --minimize-every 100 --rescale-velocities --remove-rot --rescale-every 100 --equil-steps 200
```

This performs an NVT molecular dynamics simulation at 300K for 1000 steps (0.5 ps), including performing geometry optimization, rescaling velocities, and removing rotation, both before beginning dynamics and at steps 100 and 200 of the simulation.


```shell
janus md --ensemble nve --struct tests/data/NaCl.cif --steps 200 --temp 300 --traj-start 100 --traj-every 10 --traj-file "example-trajectory.xyz" --stats-every 10 --stats-file "example-statistics.dat"
```

This performs an NVE molecular dynamics simulation at 300K for 200 steps (0.2 ps), saving the trajectory every 10 steps after the first 100, and the thermodynamical statistics every 10 steps, as well as changing the output file names for both.


For all options, run `janus md --help`.


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
