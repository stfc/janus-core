============
Installation
============

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


.. tip::

    If you are using ``uv``, ``python3 -m pip`` should be replaced with ``uv pip``


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


MLIPs requiring DGL
------------------

`DGL <https://github.com/dmlc/dgl>`_, which is a dependency of ``alignn`` and ``matgl``, no longer
publishes to PyPI, and no longer publishes any packages for Windows or MacOS.

When installing these MLIPs on Linux or MacOS, ``janus-core`` will therefore automatically install
``dgl==2.1.0``, as well as ``torch==2.2.0``, to ensure full compatibility. However, this is incompatible
with ``chgnet``, and may limit the available features in others, including ``mace``.

To use ``alignn`` and/or ``matgl`` with more recent versions of PyTorch, ``torch`` and ``dgl`` must
both be upgraded manually. Please refer to their
`installation instructions <https://www.dgl.ai/pages/start.html>`_ to upgrade ``dgl``, ensuring
that the PyTorch version, CUDA version, and OS are selected appropriately.


MLIPs with different versions of e3nn
-------------------------------------

Several MLIP packages, including ``mattersim``, ``fairchem``, and newer versions of ``sevennet``,
depend on versions of ``e3nn`` that are incompatible the version required by ``mace``. So these cannot
be installed together.


MLIPs with limited OS support
-----------------------------

Several MLIP packages have limited support on Windows We are currently unable to
support ``orb``, ``mattersim``, ``alignn`` or ``matgl`` as ``extras`` on Windows, so they
must be installed manually.
