[![Build Status][ci-badge]][ci-link]
[![Coverage Status][cov-badge]][cov-link]
[![Docs status][docs-badge]][docs-link]
[![PyPI version][pypi-badge]][pypi-link]
[![License][license-badge]][license-link]
[![DOI][doi-badge]][doi-link]

# janus-core

![logo][logo]

Tools for machine learnt interatomic potentials

## Features in development

- [x] Support for multiple MLIPs
  - MACE
  - M3GNET
  - CHGNET
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
- [x] Fine tunning MLIPs
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

> [!NOTE]
> Manually updating ASE via https://gitlab.com/ase/ase is strongly recommended, as tags may not regularly be published. For example:
> ```shell
> pip install git+https://gitlab.com/ase/ase.git@master
> ```
>
> To prevent poetry downgrading ASE when installing in future, add the repository to pyproject.toml:
>
> ```shell
> poetry add git+https://gitlab.com:ase/ase.git#master
> ```

## Examples

Tutorials for a variety of calculations can be found in the [janus-tutorials](https://github.com/stfc/janus-tutorials) repository.

### Single Point Calculations

Perform a single point calcuation (using the [MACE-MP](https://github.com/ACEsuit/mace-mp) "small" force-field):
```shell
janus singlepoint --struct tests/data/NaCl.cif --arch mace_mp --model-path small
```

This will calculate the energy, stress and forces and save this in `NaCl-results.extxyz`, in addition to generating a log file, `singlepoint.log`, and summary of inputs, `singlepoint_summary.yml`.

Additional options may be specified. For example:

```shell
janus singlepoint --struct tests/data/NaCl.cif --arch mace --model-path /path/to/your/ml.model --properties energy --properties forces --log ./example.log --out ./example.extxyz
```

This calculates both forces and energies, defines the MLIP architecture and path to your locally saved model, and changes where the log and results files are saved.

> [!NOTE]
> The MACE calculator currently returns energy, forces and stress together, so in this case the choice of property will not change the output.

By default, all structures in a trajectory file will be read, but specific structures can be selected using --read-kwargs:

```shell
janus singlepoint --struct tests/data/benzene-traj.xyz --read-kwargs "{'index': 0}"
```

For all options, run `janus singlepoint --help`.


### Geometry optimization

Perform geometry optimization (using the [MACE-MP](https://github.com/ACEsuit/mace-mp) "small" force-field):

```shell
janus geomopt --struct tests/data/H2O.cif --arch mace_mp --model-path small
```

This will optimize the atomic positions and save the resulting structure in `H2O-opt.extxyz`, in addition to generating a log file, `geomopt.log`, and summary of inputs, `geomopt_summary.yml`.

Additional options may be specified. This shares most options with `singlepoint`, as well as a few additional options, such as:

```shell
janus geomopt --struct tests/data/NaCl.cif --arch mace_mp --model-path small --vectors-only --traj 'NaCl-traj.extxyz'
```

This allows the cell vectors to be optimised, allowing only hydrostatic deformation, and saves the optimization trajectory in addition to the final structure and log.

Further options for the optimizer and filter can be specified using the `--minimize-kwargs` option. For example:

```shell
janus geomopt --struct tests/data/NaCl.cif --arch mace_mp --model-path small --fully-opt --minimize-kwargs "{'filter_kwargs': {'constant_volume' : True}, 'opt_kwargs': {'alpha': 100}}"
```

This allows the cell vectors and angles to be optimized, as well as the atomic positions, at constant volume, and sets the `alpha`, the initial guess for the Hessian, to 100 for the optimizer function.

For all options, run `janus geomopt --help`.


### Molecular dynamics

Run an NPT molecular dynamics simulation (using the [MACE-MP](https://github.com/ACEsuit/mace-mp) "small" force-field) at 300K and 1 bar for 1000 steps (1 ps):

```shell
janus md --ensemble npt --struct tests/data/NaCl.cif --arch mace_mp --model-path small --temp 300 --steps 1000 --pressure 1.0
```

This will generate several output files:

- Thermodynamical statistics every 100 steps, written to `NaCl-npt-T300.0-p1.0-stats.dat`
- The structure trajectory every 100 steps, written to `NaCl-npt-T300.0-p1.0-traj.extxyz`
- The structure to be able to restart the dynamics every 1000 steps, written to `NaCl-npt-T300.0-p1.0-res-1000.extxyz`
- The final structure written to `NaCl-npt-T300.0-p1.0-final.extxyz`
- A log of the processes carried out, written to `md.log`
- A summary of the inputs and start/end time, written to `md_summary.yml`.

Additional options may be specified. For example:

```shell
janus md --ensemble nvt --struct tests/data/NaCl.cif --steps 1000 --timestep 0.5 --temp 300 --minimize --minimize-every 100 --rescale-velocities --remove-rot --rescale-every 100 --equil-steps 200
```

This performs an NVT molecular dynamics simulation at 300K for 1000 steps (0.5 ps), including performing geometry optimization, rescaling velocities, and removing rotation, both before beginning dynamics and at steps 100 and 200 of the simulation.


```shell
janus md --ensemble nve --struct tests/data/NaCl.cif --steps 200 --temp 300 --traj-start 100 --traj-every 10 --traj-file "example-trajectory.extxyz" --stats-every 10 --stats-file "example-statistics.dat"
```

This performs an NVE molecular dynamics simulation at 300K for 200 steps (0.2 ps), saving the trajectory every 10 steps after the first 100, and the thermodynamical statistics every 10 steps, as well as changing the output file names for both.


For all options, run `janus md --help`.


### Heating

Run an NVT heating simultation from 20K to 300K in steps of 20K, with 10fs at each temperature:

```shell
janus md --ensemble nvt --struct tests/data/NaCl.cif --temp-start 20 --temp-end 300 --temp-step 20 --temp-time 10
```

The produced final, statistics, and trajectory files will indicate the heating range:

- `NaCl-nvt-T20.0-T300.0-final.extxyz`
- `NaCl-nvt-T20.0-T300.0-stats.dat`
- `NaCl-nvt-T20.0-T300.0-traj.extxyz`

The final structure file will include the final structure at each temperature point (20K, 40K, ..., 300K).

MD can also be carried out after heating using the same options as described in [Molecular dynamics](#molecular-dynamics). For example:

```shell
janus md --ensemble nvt --struct tests/data/NaCl.cif --temp-start 20 --temp-end 300 --temp-step 20 --temp-time 10 --steps 1000 --temp 300
```

This performs the same initial heating, before running a further 1000 steps (1 ps) at 300K.

When MD is run with heating, the final, trajectory, and statistics files (`NaCl-nvt-T20.0-T300.0-T300.0-final.extxyz`, `NaCl-nvt-T20.0-T300.0-T300.0-traj.extxyz`, and `NaCl-nvt-T20.0-T300.0-T300.0-stats.dat`) indicate the heating range and MD temperature, which can differ. Each file contains data from both the heating and MD parts of the simulation.

Additional settings for geometry optimization, such as enabling optimization of cell vectors by setting `hydrostatic_strain = True` for the ASE filter, can be set using the `--minimize-kwargs` option:

```shell
janus md --ensemble nvt --struct tests/data/NaCl.cif --temp-start 0 --temp-end 300 --temp-step 10 --temp-time 10 --minimize --minimize-kwargs "{'filter_kwargs': {'hydrostatic_strain' : True}}"
```

### Equation of State

Fit the equation of state for a structure (using the [MACE-MP](https://github.com/ACEsuit/mace-mp) "small" force-field):

```shell
janus eos --struct tests/data/NaCl.cif --no-minimize --min-volume 0.9 --max-volume 1.1 --n-volumes 9 --arch mace_mp --model-path small
```

This will save the energies and volumes for nine lattice constants in `NaCl-eos-raw.dat`, and the fitted minimum energy, volume, and bulk modulus in `NaCl-eos-fit.dat`, in addition to generating a log file, `eos.log`, and summary of inputs, `eos_summary.yml`.

By default, geometry optimization will be performed on the initial structure, before calculations are performed for the range of lattice constants consistent with minimum and maximum volumes supplied. Optimization at constant volume for all generated structures can also be performed (sharing the same maximum force convergence):

```shell
janus eos --struct tests/data/NaCl.cif --minimize-all --fmax 0.0001
```

For all options, run `janus eos --help`.


### Phonons

Calculate phonons with a 2x2x2 supercell, after geometry optimization (using the [MACE-MP](https://github.com/ACEsuit/mace-mp) "small" force-field):

```shell
janus phonons --struct tests/data/NaCl.cif --supercell 2x2x2 --minimize --arch mace_mp --model-path small
```

This will save the Phonopy parameters, including displacements and force constants, to `NaCl-phonopy.yml` and `NaCl-force_constants.hdf5`, in addition to generating a log file, `phonons.log`, and summary of inputs, `phonons_summary.yml`.

Additionally, the `--band` option can be added to calculate the band structure and save the results to `NaCl-auto_bands.yml`:

```shell
janus phonons --struct tests/data/NaCl.cif --supercell 2x2x2 --minimize --arch mace_mp --model-path small --band
```

If you need eigenvectors and group velocities written, add the `--write-full` option. This will generate a much larger file, but can be used to visualise phonon modes.

Further calculations, including thermal properties, DOS, and PDOS, can also be calculated (using a 2x3x4 supercell):

```shell
janus phonons --struct tests/data/NaCl.cif --supercell 2x3x4 --dos --pdos --thermal --temp-start 0 --temp-end 300 --temp-step 50
```

This will create additional output files: `NaCl-thermal.dat` for the thermal properties (heat capacity, entropy, and free energy) between 0K and 300K, `NaCl-dos.dat` for the DOS, and `NaCl-pdos.dat` for the PDOS.

For all options, run `janus phonons --help`.


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


### Training and fine-tuning MACE models

MACE models can be trained by passing a configuration file to the [MACE CLI](https://github.com/ACEsuit/mace/blob/main/mace/cli/run_train.py):

```shell
janus train --mlip-config /path/to/training/config.yml
```

This will create `logs`, `checkpoints` and `results` folders, as well as saving the trained model, and a compiled version of the model.

Foundational models can also be fine-tuned, by including the `foundation_model` option in your configuration file, and using `--fine-tune` option:

```shell
janus train --mlip-config /path/to/fine/tuning/config.yml --fine-tune
```


### Calculate MACE descriptors

MACE descriptors can be calculated for structures (using the [MACE-MP](https://github.com/ACEsuit/mace-mp) "small" force-field):

```shell
janus descriptors --struct tests/data/NaCl.cif --arch mace_mp --model-path small
```

This will calculate the mean descriptor for this structure and save this as attached information (`descriptors`) in `NaCl-descriptors.extxyz`, in addition to generating a log file, `descriptors.log`, and summary of inputs, `descriptors_summary.yml`.

The mean descriptor per element can also be calculated, and all descriptors, rather than only the invariant part, can be used when calculating the means:

```shell
janus descriptors --struct tests/data/NaCl.cif --no-invariants-only --calc-per-element
```

This will generate the same output files, but additional labels (`Cl_descriptor` and `Na_descriptor`) will be saved in `NaCl-descriptors.extxyz`.

For all options, run `janus descriptors --help`.

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
[logo]: https://raw.githubusercontent.com/stfc/janus-core/main/docs/source/images/janus-core-100.png
