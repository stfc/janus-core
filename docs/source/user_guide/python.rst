================
Python interface
================

Jupyter Notebook tutorials illustrating the use of currently available calculations can be found in the `tutorials <https://github.com/stfc/janus-core/tree/main/docs/source/tutorials>`_ documentation directory. This currently includes examples for:

.. |single_point| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/stfc/janus-core/blob/main/docs/source/tutorials/single_point.ipynb
    :alt: Launch single point Colab

.. |geom_opt| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/stfc/janus-core/blob/main/docs/source/tutorials/geom_opt.ipynb
    :alt: Launch geometry optimisation Colab

.. |md| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/stfc/janus-core/blob/main/docs/source/tutorials/md.ipynb
    :alt: Launch molecular dynamics Colab

.. |eos| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/stfc/janus-core/blob/main/docs/source/tutorials/eos.ipynb
    :alt: Launch equation of state Colab

.. |phonons| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/stfc/janus-core/blob/main/docs/source/tutorials/phonons.ipynb
    :alt: Launch phonons Colab

.. |neb| image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://colab.research.google.com/github/stfc/janus-core/blob/main/docs/source/tutorials/neb.ipynb
    :alt: Launch NEB Colab

- :doc:`Single Point </tutorials/single_point>` |single_point|
- :doc:`Geometry Optimization </tutorials/geom_opt>` |geom_opt|
- :doc:`Molecular Dynamics </tutorials/md>` |md|
- :doc:`Equation of State </tutorials/eos>` |eos|
- :doc:`Phonons </tutorials/phonons>` |phonons|
- :doc:`Nudged Elastic Band </tutorials/neb>` |neb|


Calculation outputs
===================

By default, calculations performed will modify the underlying :class:`ase.Atoms` object
to store information in the ``Atoms.info`` and ``Atoms.arrays`` dictionaries about the MLIP used.

Additional dictionary keys include ``arch``, corresponding to the MLIP architecture used,
and ``model_path``, corresponding to the model path, name or label.

Results from the MLIP calculator, which are typically stored in ``Atoms.calc.results``, will also,
by default, be copied to these dictionaries, prefixed by the MLIP ``arch``.

For example:

.. code-block:: python

    from janus_core.calculations.single_point import SinglePoint

    single_point = SinglePoint(
        struct_path="tests/data/NaCl.cif",
        arch="mace_mp",
        model_path="tests/models/mace_mp_small.model",
    )

    single_point.run()
    print(single_point.struct.info)

will return

.. code-block:: python

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

.. note::
    If running calculations with multiple MLIPs, ``arch`` and ``mlip_model`` will be overwritten with the most recent MLIP information.
    Results labelled by the architecture (e.g. ``mace_mp_energy``) will be saved between MLIPs,
    unless the same ``arch`` is chosen, in which case these values will also be overwritten.


Additional Calculators
======================

Although ``janus-core`` only directly supports the MLIP calculators listed in :doc:`Getting started </getting_started/getting_started>`,
any valid `ASE calculator <https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html>`_
can be attached to a structure, including calculators for currently unsupported MLIPs.

This structure can then be passed to ``janus-core`` calculations, which can be run as usual.

For example, performing geometry optimisation using the (`ASE built-in <https://wiki.fysik.dtu.dk/ase/ase/calculators/others.html#lennard-jones>`_) Lennard Jones potential calculator:

.. code-block:: python

    from janus_core.calculations.geom_opt import GeomOpt
    from ase.calculators.lj import LennardJones
    from ase.io import read

    struct = read("tests/data/NaCl-deformed.cif")
    struct.calc = LennardJones()

    geom_opt = GeomOpt(
        struct=struct,
        fmax=0.001,
    )
    geom_opt.run()
