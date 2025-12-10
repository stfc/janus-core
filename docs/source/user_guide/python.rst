================
Python interface
================

Jupyter Notebook tutorials illustrating the use of currently available calculations can be found in the `tutorials <https://github.com/stfc/janus-core/tree/main/docs/source/tutorials>`_ documentation directory. This currently includes examples for:

.. |single_point| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/stfc/janus-core/blob/main/docs/source/tutorials/python/single_point.ipynb
    :alt: Launch single point Colab

.. |geom_opt| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/stfc/janus-core/blob/main/docs/source/tutorials/python/geom_opt.ipynb
    :alt: Launch geometry optimisation Colab

.. |md| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/stfc/janus-core/blob/main/docs/source/tutorials/python/md.ipynb
    :alt: Launch molecular dynamics Colab

.. |eos| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/stfc/janus-core/blob/main/docs/source/tutorials/python/eos.ipynb
    :alt: Launch equation of state Colab

.. |phonons| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/stfc/janus-core/blob/main/docs/source/tutorials/python/phonons.ipynb
    :alt: Launch phonons Colab

.. |neb| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/stfc/janus-core/blob/main/docs/source/tutorials/python/neb.ipynb
    :alt: Launch NEB Colab

- :doc:`Single Point </tutorials/python/single_point>` |single_point|
- :doc:`Geometry Optimization </tutorials/python/geom_opt>` |geom_opt|
- :doc:`Molecular Dynamics </tutorials/python/md>` |md|
- :doc:`Equation of State </tutorials/python/eos>` |eos|
- :doc:`Phonons </tutorials/python/phonons>` |phonons|
- :doc:`Nudged Elastic Band </tutorials/python/neb>` |neb|

These make use of `WEAS Widget <https://weas-widget.readthedocs.io/en/latest/index.html>`_ for visualisation,
which can be installed using ``janus-core``'s ``visualise`` extra. For example:

.. code-block:: bash

    pip install janus-core[mace,visualise]

or


.. code-block:: bash

    pip install janus-core[all]


Calculation outputs
===================

By default, calculations performed will modify the underlying :class:`ase.Atoms` object
to store information in the ``Atoms.info`` and ``Atoms.arrays`` dictionaries about the MLIP used.

Additional dictionary keys include ``arch``, corresponding to the MLIP architecture used,
and ``model``, corresponding to the model path, name or label.

Results from the MLIP calculator, which are typically stored in ``Atoms.calc.results``, will also,
by default, be copied to these dictionaries, prefixed by the MLIP ``arch``.

The ``model`` and version of the MLIP package will also be saved to the calculator that
is attached to the structure in the ``Atoms.calc.parameters`` dictionary.


For example:

.. code-block:: python

    from janus_core.calculations.single_point import SinglePoint

    single_point = SinglePoint(
        struct="tests/data/NaCl.cif",
        arch="mace_mp",
        model="tests/models/mace_mp_small.model",
    )

    single_point.run()
    print(single_point.struct.info)
    print(single_point.struct.calc.parameters)

will return

.. code-block:: python

    {
        'spacegroup': Spacegroup(1, setting=1),
        'unit_cell': 'conventional',
        'occupancy': {'0': {'Na': 1.0}, '1': {'Cl': 1.0}, '2': {'Na': 1.0}, '3': {'Cl': 1.0}, '4': {'Na': 1.0}, '5': {'Cl': 1.0}, '6': {'Na': 1.0}, '7': {'Cl': 1.0}},
        'model': 'tests/models/mace_mp_small.model',
        'arch': 'mace_mp',
        'mace_mp_energy': -27.035127799332745,
        'mace_mp_stress': array([-4.78327600e-03, -4.78327600e-03, -4.78327600e-03,  1.08000967e-19, -2.74004242e-19, -2.04504710e-19]),
        'system_name': 'NaCl',
    }

    {'version': '0.3.14', 'arch': 'mace_mp', 'model': 'tests/models/mace_mp_small.model'}


.. note::
    If running calculations with multiple MLIPs, ``arch`` and ``mlip_model`` will be overwritten with the most recent MLIP information.
    Results labelled by the architecture (e.g. ``mace_mp_energy``) will be saved between MLIPs,
    unless the same ``arch`` is chosen, in which case these values will also be overwritten.


D3 Dispersion
=============

A PyTorch implementation of DFTD2 and DFTD3, using the `TorchDFTD3Calculator <https://github.com/pfnet-research/torch-dftd>`_,
can be used to add dispersion corrections to MLIP predictions.

The required Python pacakge is included with ``mace_mp``, but can also be installed as its own extra:

.. code-block:: bash

    pip install janus-core[d3]


Once installed, dispersion can be added through ``calc_kwargs`` through the ``dispersion`` keyword,
with ``dispersion_kwargs`` used to pass any further keywords to the ``TorchDFTD3Calculator``:

.. code-block:: python

    from ase import units

    from janus_core.calculations.single_point import SinglePoint

    single_point = SinglePoint(
        struct="tests/data/NaCl.cif",
        arch="mace_mp",
        model="tests/models/mace_mp_small.model",
        calc_kwargs={"dispersion": True, "dispersion_kwargs": {"cutoff":  95.0 * units.Bohr}}
    )

.. note::
    In most cases, defaults for ``dispersion_kwargs`` are those set within ``TorchDFTD3Calculator``,
    but in the case of ``mace_mp``, we mirror the corresponding defaults from the
    ``mace.calculators.mace_mp`` function.


The ``TorchDFTD3Calculator`` can also be added to any existing calculator if required:

.. note::
    Keyword arguments for ``TorchDFTD3Calculator`` should be passed directly here,
    as shown with ``cutoff``. This will not have access to ``mace_mp`` default values,
    so will always use defaults from ``TorchDFTD3Calculator``.


.. code-block:: python

    from ase import units
    from ase.io import read

    from janus_core.calculations.single_point import SinglePoint
    from janus_core.helpers.mlip_calculators import add_dispersion, choose_calculator

    struct = read("tests/data/NaCl-deformed.cif")

    mace_calc = choose_calculator("mace_mp")
    calc = add_dispersion(mace_calc, device="cpu", cutoff=95 * units.Bohr)

    struct.calc = calc

    single_point = SinglePoint(struct=struct)
    single_point.run()


Additional Calculators
======================

Although ``janus-core`` only directly supports the MLIP calculators listed in :doc:`Getting started </user_guide/get_started>`,
any valid `ASE calculator <https://ase-lib.org/ase/calculators/calculators.html>`_
can be attached to a structure, including calculators for currently unsupported MLIPs.

This structure can then be passed to ``janus-core`` calculations, which can be run as usual.

For example, performing geometry optimisation using the (`ASE built-in <https://ase-lib.org/ase/calculators/others.html#lennard-jones>`_) Lennard Jones potential calculator:

.. code-block:: python

    from ase.calculators.lj import LennardJones
    from ase.io import read

    from janus_core.calculations.geom_opt import GeomOpt

    struct = read("tests/data/NaCl-deformed.cif")
    struct.calc = LennardJones()

    geom_opt = GeomOpt(
        struct=struct,
        fmax=0.001,
    )
    geom_opt.run()


Similarly, if you have any issues setting up a calculator from a supported MLIP, these
can still be set up manually:

.. code-block:: python

    from ase.io import read
    from mace.calculators import mace_mp

    from janus_core.calculations.geom_opt import GeomOpt

    struct = read("tests/data/NaCl-deformed.cif")
    struct.calc = mace_mp()

    geom_opt = GeomOpt(
        struct=struct,
        fmax=0.001,
    )
    geom_opt.run()

.. note::
    Setting up a calculator this way means that we are unable to set the ``Atoms.info``
    details about the calculator, such as ``model`` and ``arch``, or
    ``Atoms.calc.parameters``, described in `Calculation outputs`_.
