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

    ``matgl`` and ``alignn`` depend on `dgl <https://github.com/dmlc/dgl?tab=readme-ov-file>`_,
    which no longer publishes to PyPI. If ``janus-core`` is installed with either of these extras,
    PyTorch will automatically be set to 2.2.0 to ensure compatibility. However, this is incompatible
    with ``chgnet``, and may limit the available features in others, including ``mace``. To use
    ``matgl`` and/or ``alignn`` with more recent PyTorch release, please refer to the
    :doc:`installation documentation </user_guide/installation>`.

To install all MLIPs that do not depend on ``dgl``:

.. code-block:: python

    python3 -m pip install janus-core[all]

Currently supported extras are:

- ``alignn``: `ALIGNN <https://github.com/usnistgov/alignn>`_
- ``chgnet``: `CHGNet <https://github.com/CederGroupHub/chgnet/>`_
- ``mace``: `MACE <https://github.com/ACEsuit/mace>`_
- ``m3gnet``: `M3GNet <https://github.com/materialsvirtuallab/matgl/>`_
- ``sevenn``: `SevenNet <https://github.com/MDIL-SNU/SevenNet/>`_
- ``nequip``: `NequIP <https://github.com/mir-group/nequip>`_
- ``dpa3``: `DPA3 <https://github.com/deepmodeling/deepmd-kit/tree/dpa3-alpha>`_
- ``orb``: `Orb <https://github.com/orbital-materials/orb-models>`_
- ``mattersim``: `MatterSim <https://github.com/microsoft/mattersim>`_

.. note::

    ``orb`` and ``mattersim`` are not currently compatible with Windows natively,
    but can be installed and run via Windows Subsystem for Linux.


``extras`` are also listed in `pyproject.toml <https://github.com/stfc/janus-core/blob/main/pyproject.toml>`_ under ``[project.optional-dependencies]``.
