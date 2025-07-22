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


To install all MLIPs and features currently compatible with MACE, run:

.. code-block:: python

    python3 -m pip install janus-core[all]


Currently supported MLIP ``extras`` are:

- ``alignn``: `ALIGNN <https://github.com/usnistgov/alignn>`_
- ``chgnet``: `CHGNet <https://github.com/CederGroupHub/chgnet/>`_
- ``mace``: `MACE <https://github.com/ACEsuit/mace>`_
- ``m3gnet``: `M3GNet <https://github.com/materialsvirtuallab/matgl/>`_
- ``sevenn``: `SevenNet <https://github.com/MDIL-SNU/SevenNet/>`_
- ``nequip``: `NequIP <https://github.com/mir-group/nequip>`_
- ``dpa3``: `DPA3 <https://github.com/deepmodeling/deepmd-kit/tree/dpa3-alpha>`_
- ``orb``: `Orb <https://github.com/orbital-materials/orb-models>`_
- ``mattersim``: `MatterSim <https://github.com/microsoft/mattersim>`_
- ``grace``: `GRACE <https://github.com/ICAMS/grace-tensorpotential>`_
- ``fairchem``: `eqV2 DeNS/eSEN <https://github.com/FAIR-Chem/fairchem/tree/main/src/fairchem/core>`_

.. note::

    ``orb`` and ``mattersim`` are not currently compatible with Windows natively,
    but can be installed and run via Windows Subsystem for Linux.


Additional features can also be enabled as ``extras``:

- ``d3``: `DFTD3 <https://github.com/pfnet-research/torch-dftd>`_
- ``visualise``: `WEAS Widget <https://github.com/superstar54/weas-widget>`_
- ``plumed``: `PLUMED <https://www.plumed.org>`_

.. note::

    PLUMED requires further installation steps, described in the :doc:`installation documentation </user_guide/installation/>`.


``extras`` are also listed in `pyproject.toml <https://github.com/stfc/janus-core/blob/main/pyproject.toml>`_ under ``[project.optional-dependencies]``.
