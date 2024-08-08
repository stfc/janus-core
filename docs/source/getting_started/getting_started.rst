===============
Getting started
===============

Dependencies
------------

``janus-core`` dependencies currently include:

- Python >= 3.9
- ASE >= 3.23
- chgnet = 0.3.8
- mace-torch = 0.3.6
- matgl = 1.1.2
- sevenn = 0.9.3 (optional)
- alignn = 2024.5.27 (optional)

All required and optional dependencies can be found in `pyproject.toml <https://github.com/stfc/janus-core/blob/main/pyproject.toml>`_.

.. note::
    Where possible, we expect to update pinned MLIP dependencies to match their latest releases, subject to any required API fixes.


Installation
------------

The latest stable release of ``janus-core``, including its dependencies, can be installed from PyPI by running:

.. code-block:: bash

    python3 -m pip install janus-core

To get all the latest changes, ``janus-core`` can also be installed from GitHub:

.. code-block:: bash

    python3 -m pip install git+https://github.com/stfc/janus-core.git
