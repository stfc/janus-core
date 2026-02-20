============
Installation
============

Advanced installation
=====================

While `pip <https://packaging.python.org/en/latest/guides/tool-recommendations/#installing-packages>`_
is a quick and accessible way to get started, we strongly recommend installing a package manager,
such as `uv <https://docs.astral.sh/uv/>`_ or `micromamba <https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html>`_
to install ``janus-core`` and its dependencies for practical use.

This guide focuses on ``uv``, but many package managers provide similar benefits, including installing and managing Python itself
(particularly useful in high performance computing environments), much faster dependency installation,
and virtual environment management.

In particular, we will describe the use of ``uv pip``, which is designed to resemble the standard ``pip`` interface,
with similar commands (``uv pip install``,  ``uv pip list``, ``uv pip tree``, etc.).
`Compared with pip <https://docs.astral.sh/uv/pip/compatibility/>`_, ``uv pip`` is slightly lower level
and tends to be stricter, but in most cases ``uv pip`` can be used as a drop-in replacement for ``pip``.

1. `Install uv <https://docs.astral.sh/uv/getting-started/installation/>`_

2. Create and activate your virtual environment, specifying the desired Python version:

.. code-block:: bash

    uv venv -p 3.12
    source .venv/bin/activate

.. tip::

    If you have already created a virtual environment using ``python3 -m venv``,
    ``micromamba``, ``conda``, etc, ``uv`` can still install dependencies into it if
    this is activated instead.


3. Use ``uv pip`` in place of ``python3 -m pip`` to install ``janus-core``
and any optional dependencies (in this case, MACE), into your virtual environment:

.. code-block:: bash

    uv pip install janus-core[mace] -U


.. note::

    We continue to refer to ``python3 -m pip`` elsewhere in the documentation for generality.


PLUMED
======

The `Plumed Python wrapper <https://pypi.org/project/plumed/>`_ is installable as an ``extra``:

.. code-block:: bash

    python3 -m pip install janus-core[plumed]


However, as described in the linked package description, it is necessary to additionally build and install
the PLUMED library, and set the ``PLUMED_KERNEL`` environment variable.

Further installation options are described in the `documentation <https://www.plumed.org/download>`_.

.. warning::

    Pre-compiled binary sources, including those available through ``conda-forge``, may not correspond to the latest release.
    For example, there are resolved issues on MacOS that require building from source.


Updating Dependencies
=====================

Installing from PyPI
--------------------

When installing ``janus-core``, dependencies are automatically selected to be consistent with all
``extras``.

For an individual MLIP, you may be able to upgrade dependencies beyond these defaults,
to include newer features or bug fixes. For example, to upgrade PyTorch to the latest version:

.. code-block:: bash

    python3 -m pip install -U torch


or a specific version:

.. code-block:: bash

    python3 -m pip install torch==2.5.1


Installing from git repositories
--------------------------------

It may be useful to upgrade dependencies to include their latest changes, or code in development,
often available on GitHub or GitLab.

For example, to use the latest version of ASE:

.. code-block:: bash

    python3 -m pip install git+https://gitlab.com/ase/ase.git

Specific branches may also be targeted:

.. code-block:: bash

    python3 -m pip install git+https://gitlab.com/drFaustroll/ase.git@npt_triangular

This will install the ``npt_triangular`` branch of https://gitlab.com/drFaustroll/ase, which includes a
fix to allow NPT when the computational box is not an upper triangular matrix.


Additional libraries
--------------------

Some libraries are not installed by default, but may improve performance, such as:

- ``cuEquivariance`` can be used for `CUDA accerlation of MACE <https://mace-docs.readthedocs.io/en/latest/guide/cuda_acceleration.html>`_ (with PyTorch 2.4 onwards)
- `PyTorch implementation of DFTD3 <https://github.com/CheukHinHoJerry/torch-dftd.git>`_, which can be used by MACE calculations on GPU


MLIP Incompatibilies
====================

Due to the different requirements of the MLIPs we support, it is not always possible to install all combinations of ``extras``.


MLIPs with different versions of e3nn
-------------------------------------

Several MLIP packages, including ``mattersim``, ``fairchem``, and newer versions of ``sevennet``,
depend on versions of ``e3nn`` that are incompatible the version required by ``mace``. So these cannot
be installed together.


MLIPs with limited OS support
-----------------------------

Several MLIP packages have limited support on Windows. We are currently unable to
support ``orb``, ``mattersim``, or ``pet-mad`` as ``extras`` on Windows, so they
must be installed manually.
