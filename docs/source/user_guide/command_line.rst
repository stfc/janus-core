======================
Command line interface
======================

Basic usage
-----------

General help, including a description of all available commands, can be found by running:

.. code-block:: bash

    janus --help

This should return something similar to:

.. code-block:: bash

    Usage: janus [OPTIONS] COMMAND [ARGS]...

    Options:
      --version                       Print janus version and exit.
      --install-completion [bash|zsh|fish|powershell|pwsh]
                                      Install completion for the specified shell.
      --show-completion [bash|zsh|fish|powershell|pwsh]
                                      Show completion for the specified shell, to
                                      copy it or customize the installation.
      --help                          Show this message and exit.

    Commands:
      descriptors  Calculate MLIP descriptors.
      eos          Calculate equation of state.
      geomopt      Perform geometry optimization and save optimized structure...
      md           Run molecular dynamics simulation, and save trajectory and...
      phonons      Calculate phonons and save results.
      singlepoint  Perform single point calculations and save to file.
      train        Running training for an MLIP.

.. tip::

   Auto completion for your shell can be set up by running ``janus --install-completion``

A description of each subcommand, as well as valid options, can be listed b adding the ``--help`` option after the subcommand. For example,

.. code-block:: bash

    janus singlepoint --help


prints the following:

.. code-block:: bash

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


Using configuration files
-------------------------

Default values for all command line options may be specifed through a Yaml 1.1 formatted configuration file by adding the ``--config`` option.
If an option is present in both the command line and configuration file, the command line value takes precedence.

For example, with the following configuration file and command:

.. code-block:: yaml

    struct: "NaCl.cif"
    properties:
    - "energy"
    out: "NaCl-results.extxyz"
    arch: mace_mp
    model-path: medium
    calc-kwargs:
      dispersion: True


.. code-block:: bash

    janus singlepoint --struct KCl.cif --out KCl-results.cif --config config.yml


This will run a singlepoint energy calculation on ``KCl.cif`` using the `MACE-MP <https://github.com/ACEsuit/mace-mp>`_ "medium" force-field, saving the results to ``KCl-results.cif``.


.. note::
    ``properties`` must be passed as a Yaml list, as above, not as a string.

Example configurations for all commands can be found in `janus-tutorials <https://github.com/stfc/janus-tutorials/tree/main/configs>`_


Single point calculations
-------------------------

Perform a single point calcuation (using the `MACE-MP <https://github.com/ACEsuit/mace-mp>`_ "small" force-field):

.. code-block:: bash

    janus singlepoint --struct tests/data/NaCl.cif --arch mace_mp --model-path small


This will calculate the energy, stress and forces and save this in ``NaCl-results.extxyz``, in addition to generating a log file, ``NaCl-singlepoint-log.yml``, and summary of inputs, ``NaCl-singlepoint-summary.yml``.

Additional options may be specified. For example:

.. code-block:: bash

    janus singlepoint --struct tests/data/NaCl.cif --arch mace --model-path /path/to/your/ml.model --properties energy --properties forces --log ./example.log --out ./example.extxyz


This calculates both forces and energies, defines the MLIP architecture and path to your locally saved model, and changes where the log and results files are saved.

.. note::

    The MACE calculator currently returns energy, forces and stress together, so in this case the choice of property will not change the output.

By default, all structures in a trajectory file will be read, but specific structures can be selected using --read-kwargs:

.. code-block:: bash

    janus singlepoint --struct tests/data/benzene-traj.xyz --read-kwargs "{'index': 0}"


For all options, run ``janus singlepoint --help``.


Geometry optimization
---------------------

Perform geometry optimization (using the `MACE-MP <https://github.com/ACEsuit/mace-mp>`_ "small" force-field):

.. code-block:: bash

    janus geomopt --struct tests/data/H2O.cif --arch mace_mp --model-path small


This will optimize the atomic positions and save the resulting structure in ``H2O-opt.extxyz``, in addition to generating a log file, ``H20-geomopt-log.yml``, and summary of inputs, ``H20-geomopt-summary.yml``.

Additional options may be specified. This shares most options with ``singlepoint``, as well as a few additional options, such as:

.. code-block:: bash

    janus geomopt --struct tests/data/NaCl.cif --arch mace_mp --model-path small --opt-cell-lengths --traj 'NaCl-traj.extxyz'


This allows the cell vectors to be optimised, allowing only hydrostatic deformation, and saves the optimization trajectory in addition to the final structure and log.

Further options for the optimizer and filter can be specified using the ``--minimize-kwargs`` option. For example:

.. code-block:: bash

    janus geomopt --struct tests/data/NaCl.cif --arch mace_mp --model-path small --opt-cell-fully --minimize-kwargs "{'filter_kwargs': {'constant_volume' : True}, 'opt_kwargs': {'alpha': 100}}"


This allows the cell vectors and angles to be optimized, as well as the atomic positions, at constant volume, and sets the ``alpha``, the initial guess for the Hessian, to 100 for the optimizer function.

For all options, run ``janus geomopt --help``.


Molecular dynamics
------------------

Run an NPT molecular dynamics simulation (using the `MACE-MP <https://github.com/ACEsuit/mace-mp>`_ "small" force-field) at 300K and 1 bar for 1000 steps (1 ps, default time-step is 1 fs):

.. code-block:: bash

    janus md --ensemble npt --struct tests/data/NaCl.cif --arch mace_mp --model-path small --temp 300 --steps 1000 --pressure 1.0


This will generate several output files:

- Thermodynamical statistics every 100 steps, written to ``NaCl-npt-T300.0-p1.0-stats.dat``
- The structure trajectory every 100 steps, written to ``NaCl-npt-T300.0-p1.0-traj.extxyz``
- The structure to be able to restart the dynamics every 1000 steps, written to ``NaCl-npt-T300.0-p1.0-res-1000.extxyz``
- The final structure written to ``NaCl-npt-T300.0-p1.0-final.extxyz``
- A log of the processes carried out, written to ``NaCl-npt-T300.0-p1.0-md-log.yml``
- A summary of the inputs and start/end time, written to ``NaCl-npt-T300.0-p1.0-md-summary.yml``.

Additional options may be specified. For example:

.. code-block:: bash

    janus md --ensemble nvt --struct tests/data/NaCl.cif --steps 1000 --timestep 0.5 --temp 300 --minimize --minimize-every 100 --rescale-velocities --remove-rot --rescale-every 100 --equil-steps 200


This performs an NVT molecular dynamics simulation at 300K for 1000 steps (0.5 ps), including performing geometry optimization, rescaling velocities, and removing rotation,
both before beginning dynamics and at steps 100 and 200 of the simulation.


.. code-block:: bash

    janus md --ensemble nve --struct tests/data/NaCl.cif --steps 200 --temp 300 --traj-start 100 --traj-every 10 --traj-file "example-trajectory.extxyz" --stats-every 10 --stats-file "example-statistics.dat"


This performs an NVE molecular dynamics simulation at 300K for 200 steps (0.2 ps), saving the trajectory every 10 steps after the first 100, and the thermodynamical statistics every 10 steps,
as well as changing the output file names for both.

To restart a simulation, the restart file can be entered explicitly (e.g. ``--struct NaCl-npt-T300.0-p1.0-res-1000.extxyz``) in combination with the ``--restart`` option,
but by default the ``--restart-auto`` option allows the same original structure to be used to infer the most recent restart file created.
For example:

.. code-block:: bash

    janus md --ensemble nvt --struct tests/data/NaCl.cif --steps 1200 --stats-every 10 --traj-every 100 --restart-every 1000
    janus md --ensemble nvt --struct tests/data/NaCl.cif --steps 1200 --stats-every 10 --traj-every 100 --restart

will create, then restart from, ``NaCl-npt-T300.0-p1.0-res-1000.extxyz``,
running an additional 1000 steps, and appending the statistics and trajectory files created by the first command.


.. note::
    Depending on the frequency of statistics, trajectory, and restart file outputs, it is likely that outputs for some steps may be repeated.
    These can be identified using the ``Step`` column in the statistics file, and the ``step`` info label for each trajectory image.
    For example, in the above example, outputs between steps 1000 and 1200 would be repeated.


For all options, run ``janus md --help``.


Heating
-------

Run an NVT heating simultation from 20K to 300K in steps of 20K, with 10fs at each temperature:

.. code-block:: bash

    janus md --ensemble nvt --struct tests/data/NaCl.cif --temp-start 20 --temp-end 300 --temp-step 20 --temp-time 10


The produced final, statistics, and trajectory files will indicate the heating range:

- ``NaCl-nvt-T20.0-T300.0-final.extxyz``
- ``NaCl-nvt-T20.0-T300.0-stats.dat``
- ``NaCl-nvt-T20.0-T300.0-traj.extxyz``

The final structure file will include the final structure at each temperature point (20K, 40K, ..., 300K).

MD can also be carried out after heating using the same options as described in `Molecular dynamics`_. For example:

.. code-block:: bash

    janus md --ensemble nvt --struct tests/data/NaCl.cif --temp-start 20 --temp-end 300 --temp-step 20 --temp-time 10 --steps 1000 --temp 300


This performs the same initial heating, before running a further 1000 steps (1 ps) at 300K.

When MD is run with heating, the final, trajectory, and statistics files (``NaCl-nvt-T20.0-T300.0-T300.0-final.extxyz``, ``NaCl-nvt-T20.0-T300.0-T300.0-traj.extxyz``, and ``NaCl-nvt-T20.0-T300.0-T300.0-stats.dat``)
indicate the heating range and MD temperature, which can differ. Each file contains data from both the heating and MD parts of the simulation.

Additional settings for geometry optimization, such as enabling optimization of cell vectors by setting ``hydrostatic_strain = True`` for the ASE filter, can be set using the ``--minimize-kwargs`` option:

.. code-block:: bash

    janus md --ensemble nvt --struct tests/data/NaCl.cif --temp-start 0 --temp-end 300 --temp-step 10 --temp-time 10 --minimize --minimize-kwargs "{'filter_kwargs': {'hydrostatic_strain' : True}}"


Equation of State
-----------------

Fit the equation of state for a structure (using the `MACE-MP <https://github.com/ACEsuit/mace-mp>`_ "small" force-field):

.. code-block:: bash

    janus eos --struct tests/data/NaCl.cif --no-minimize --min-volume 0.9 --max-volume 1.1 --n-volumes 9 --arch mace_mp --model-path small


This will save the energies and volumes for nine lattice constants in ``NaCl-eos-raw.dat``, and the fitted minimum energy, volume, and bulk modulus in ``NaCl-eos-fit.dat``,
in addition to generating a log file, ``NaCl-eos-log.yml``, and summary of inputs, ``NaCl-eos-summary.yml``.

By default, geometry optimization will be performed on the initial structure, before calculations are performed for the range of lattice constants consistent with minimum and maximum volumes supplied.
Optimization at constant volume for all generated structures can also be performed (sharing the same maximum force convergence):

.. code-block:: bash

    janus eos --struct tests/data/NaCl.cif --minimize-all --fmax 0.0001


For all options, run ``janus eos --help``.


Phonons
-------

Calculate phonons with a 2x2x2 supercell, after geometry optimization (using the `MACE-MP <https://github.com/ACEsuit/mace-mp>`_ "small" force-field):

.. code-block:: bash

    janus phonons --struct tests/data/NaCl.cif --supercell "2 2 2" --minimize --arch mace_mp --model-path small


This will save the Phonopy parameters, including displacements and force constants, to ``NaCl-phonopy.yml`` and ``NaCl-force_constants.hdf5``,
in addition to generating a log file, ``NaCl-phonons-log.yml``, and summary of inputs, ``NaCl-phonons-summary.yml``.

Additionally, the ``--bands`` option can be added to calculate the band structure and save the results to ``NaCl-auto_bands.yml``:

.. code-block:: bash

    janus phonons --struct tests/data/NaCl.cif --supercell "2 2 2" --minimize --arch mace_mp --model-path small --bands


If you need eigenvectors and group velocities written, add the ``--write-full`` option. This will generate a much larger file, but can be used to visualise phonon modes.

Further calculations, including thermal properties, DOS, and PDOS, can also be calculated (using a 2x3x4 supercell):

.. code-block:: bash

    janus phonons --struct tests/data/NaCl.cif --supercell "2 3 4" --dos --pdos --thermal --temp-start 0 --temp-end 300 --temp-step 50


This will create additional output files: ``NaCl-thermal.dat`` for the thermal properties (heat capacity, entropy, and free energy)
between 0K and 300K, ``NaCl-dos.dat`` for the DOS, and ``NaCl-pdos.dat`` for the PDOS.

To define the supercell, the ``--supercell`` option can be used, which *must* be passed in as a space-separated string.
Similar to Phonopy, the supercell matrix can be defined in three ways:

1. One integer (``--supercell "2"``) specifying all diagonal elements.
2. Three integers (``--supercell "2 2 2"``) specifying each individual diagonal element.
3. Nine integers (``--supercell "2 0 0 0 2 0 0 0 2"``) specifying all elements, filling the matrix row-wise.

For all options, run ``janus phonons --help``.


Training and fine-tuning MLIPs
------------------------------

.. note::
    Currently only MACE models are supported. See the `MACE CLI <https://github.com/ACEsuit/mace/blob/main/mace/cli/run_train.py>`_ for further configuration details

Models can be trained by passing a configuration file to the MLIP's command line interface:

.. code-block:: bash

    janus train --mlip-config /path/to/training/config.yml

For MACE, this will create ``logs``, ``checkpoints`` and ``results`` directories, as well as saving the trained model, and a compiled version of the model.
Additionally, a log file, ``train-log.yml``, and summary file, ``train-summary.yml``, will be generated.

Foundational models can also be fine-tuned, by including the ``foundation_model`` option in your configuration file, and using ``--fine-tune`` option:

.. code-block:: bash

    janus train --mlip-config /path/to/fine/tuning/config.yml --fine-tune


Calculate descriptors
---------------------

.. note::
    Currently only MACE models are supported for this calculation

Descriptors of a structure can be calculated (using the `MACE-MP <https://github.com/ACEsuit/mace-mp>`_ "small" force-field):

.. code-block:: bash

    janus descriptors --struct tests/data/NaCl.cif --arch mace_mp --model-path small


This will calculate the mean descriptor for this structure and save this as attached information (``mace_mp_descriptors``) in ``NaCl-descriptors.extxyz``,
in addition to generating a log file, ``NaCl-descriptors-log.yml``, and summary of inputs, ``NaCl-descriptors-summary.yml``.

The mean descriptor per element can also be calculated, and all descriptors, rather than only the invariant part, can be used when calculating the means:

.. code-block:: bash

    janus descriptors --struct tests/data/NaCl.cif --no-invariants-only --calc-per-element


This will generate the same output files, but additional labels (``mace_mp_Cl_descriptor`` and ``mace_mp_Na_descriptor``) will be saved in ``NaCl-descriptors.extxyz``.

For all options, run ``janus descriptors --help``.
