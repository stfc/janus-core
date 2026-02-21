===============
Getting started
===============

Dependencies
------------

All required and optional dependencies can be found in `pyproject.toml <https://github.com/stfc/janus-core/blob/main/pyproject.toml>`_.


Installation
------------

The latest stable release of ``janus-core``, including its dependencies, can be installed from PyPI by running:

.. code-block:: bash

    python3 -m pip install janus-core


To get all the latest changes, ``janus-core`` can also be installed from GitHub:

.. code-block:: bash

    python3 -m pip install git+https://github.com/stfc/janus-core.git


By default, no machine learnt interatomic potentials (MLIPs) will be installed. These can be installed manually, or as ``extras`` with ``janus-core``.

For example, to install MACE, CHGNet, and SevenNet, run:

.. code-block:: python

    python3 -m pip install janus-core[chgnet,sevennet]


.. warning::

    We are unable to support for automatic installation of all combinations of MLIPs, or MLIPs on all platforms.
    Please refer to the :doc:`installation documentation </user_guide/installation>` for more details.


Currently supported MLIP ``extras`` are:

- ``chgnet``: `CHGNet (chgnet) <https://github.com/CederGroupHub/chgnet/>`_
- ``mace``: `MACE (mace, mace_mp, mace_off, mace_omol) <https://github.com/ACEsuit/mace>`_
- ``sevenn``: `SevenNet (sevennet) <https://github.com/MDIL-SNU/SevenNet/>`_
- ``nequip``: `NequIP (nequip) <https://github.com/mir-group/nequip>`_
- ``dpa3``: `DPA3 (dpa3) <https://github.com/deepmodeling/deepmd-kit/tree/dpa3-alpha>`_
- ``orb``: `Orb (orb) <https://github.com/orbital-materials/orb-models>`_
- ``mattersim``: `MatterSim (mattersim) <https://github.com/microsoft/mattersim>`_
- ``grace``: `GRACE (grace) <https://github.com/ICAMS/grace-tensorpotential>`_
- ``fairchem``: `UMA (uma) <https://github.com/FAIR-Chem/fairchem/tree/main/src/fairchem/core>`_
- ``pet-mad``: `PET-MAD (pet_mad) <https://github.com/lab-cosmo/pet-mad>`_

The labels in brackets are the corresponding architecture parameters (``arch``) that
should be set to use these models.

.. note::

    ``orb``, ``mattersim``, and ``pet-mad`` are not currently compatible with Windows
    natively, but can be installed and run via Windows Subsystem for Linux.


Additional features can also be enabled as ``extras``:

- ``d3``: `DFTD3 <https://github.com/pfnet-research/torch-dftd>`_
- ``visualise``: `WEAS Widget <https://github.com/superstar54/weas-widget>`_
- ``plumed``: `PLUMED <https://www.plumed.org>`_

.. note::

    PLUMED requires further installation steps, described in the :doc:`installation documentation </user_guide/installation/>`.


``extras`` are also listed in `pyproject.toml <https://github.com/stfc/janus-core/blob/main/pyproject.toml>`_ under ``[project.optional-dependencies]``.
