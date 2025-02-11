===============
Getting started
===============

Dependencies
------------

``janus-core`` dependencies currently include:

- Python >= 3.10
- ASE >= 3.24
- mace-torch = 0.3.10
- chgnet = 0.4.0 (optional)
- matgl = 1.1.3 (optional)
- sevenn = 0.10.3 (optional)
- alignn = 2024.5.27 (optional)
- nequip = 0.6.1 (optional)
- deepmd-kit = dpa3-alpha (optional)
- orb-models = 0.4.2 (optional)

Where possible, we aim to update pinned MLIP dependencies to match their latest releases, subject to any required API fixes.

All required and optional dependencies can be found in `pyproject.toml <https://github.com/stfc/janus-core/blob/main/pyproject.toml>`_.


Installation
------------

The latest stable release of ``janus-core``, including its dependencies, can be installed from PyPI by running:

.. code-block:: bash

    python3 -m pip install janus-core

To get all the latest changes, ``janus-core`` can also be installed from GitHub:

.. code-block:: bash

    python3 -m pip install git+https://github.com/stfc/janus-core.git

By default, MACE is the only MLIP installed.

Other MLIPs can be installed as ``extras``. For example, to install CHGNet and SevenNet, run:

.. code-block:: python

    python3 -m pip install janus-core[chgnet,sevennet]


.. warning::

    ``matgl`` and ``alignn`` depend on `dgl <https://github.com/dmlc/dgl?tab=readme-ov-file>`_,
    which no longer publishes to PyPI. If ``janus-core`` is installed with either of these extras,
    PyTorch will automatically be set to 2.2.0 to ensure compatibility. However, this is incompatible
    with ``chgnet``, and may limit the available features in others, including ``mace``. To use
    ``matgl`` and/or ``alignn`` with more recent PyTorch release, please refer to the
    :doc:`installation documentation </user_guide/installation>`.

To install all MLIPs that do not depend on `dgl`:

.. code-block:: python

    python3 -m pip install janus-core[all]

Currently supported extras are:

- ``alignn``: `ALIGNN <https://github.com/usnistgov/alignn>`_
- ``chgnet``: `CHGNet <https://github.com/CederGroupHub/chgnet/>`_
- ``m3gnet``: `M3GNet <https://github.com/materialsvirtuallab/matgl/>`_
- ``sevenn``: `SevenNet <https://github.com/MDIL-SNU/SevenNet/>`_
- ``nequip``: `NequIP <https://github.com/mir-group/nequip>`_
- ``dpa3``: `DPA3 <https://github.com/deepmodeling/deepmd-kit/tree/dpa3-alpha>`_
- ``orb``: `Orb <https://github.com/orbital-materials/orb-models>`_

``extras`` are also listed in `pyproject.toml <https://github.com/stfc/janus-core/blob/main/pyproject.toml>`_ under ``[project.optional-dependencies]``.
