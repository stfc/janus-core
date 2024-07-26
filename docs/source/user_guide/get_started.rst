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

All required and optional dependencies can be found in `pyproject.toml <https://github.com/stfc/janus-core/blob/main/pyproject.toml>`_.

.. note::
    Where possible, we expect to update pinned MLIP dependencies to match their latest releases, subject to any required API fixes.

.. note::
    Manually updating ASE to include the latest commits is strongly recommended, as tags may not regularly be published. For example:

    .. code-block:: bash

        pip install git+https://gitlab.com/ase/ase.git@master

    When using poetry, ``pyproject.toml`` can be modified to prevent ASE being downgraded when installing in future by running:

    .. code-block:: bash

        poetry add git+https://gitlab.com:ase/ase.git#master


Installation
------------

The latest release of ``janus-core``, including its dependencies, can be installed from PyPI by running:

.. code-block:: bash

    pip install janus-core

To download the latest changes, ``janus-core`` can also be installed from source:

.. code-block:: bash

    pip install git+https://github.com/stfc/janus-core.git
