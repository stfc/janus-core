===============
Getting started
===============

We recommend `installing uv <https://docs.astral.sh/uv/getting-started/installation/>`_ for dependency management when developing for ``janus-core``.

This provides a number of useful features, including:

- `Dependency management <https://docs.astral.sh/uv/concepts/projects/dependencies/>`_ (``uv [add,remove]`` etc.) and organization (`groups <https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-groups>`_)

- Storing the versions of all installations in a `uv.lock <https://docs.astral.sh/uv/concepts/projects/sync/>`_ file, for reproducible builds

- Improved `dependency resolution <https://docs.astral.sh/uv/concepts/resolution/>`_

- Virtual environment management

- `Building and publishing <https://docs.astral.sh/uv/guides/publish/>`_ tools

  * Currently, an external build backend, such as `pdm <https://pypi.org/project/pdm-backend>`_, is required


Dependencies useful for development can then be installed by running::

    uv sync
    source .venv/bin/activate


Extras, such as optional MLIPs, can also be installed by running::

    uv sync --extra alignn --extra sevennet

or to install all supported MLIPs::

    uv sync --all-extras


Using uv
++++++++

``uv`` manages a `persistent environment <https://docs.astral.sh/uv/concepts/projects/layout/#the-project-environment>`_
with the project and its dependencies in a ``.venv`` directory, adjacent to ``pyproject.toml``. This will be created automatically as needed.

``uv`` provides two separate APIs for managing your Python project and environment.

``uv pip`` is designed to resemble the ``pip`` CLI, with similar commands (``uv pip install``,  ``uv pip list``, ``uv pip tree``, etc.),
and is slightly lower level. `Compared with pip <https://docs.astral.sh/uv/pip/compatibility/>`_,
``uv`` tends to be stricter, but in most cases ``uv pip`` could be used in place of ``pip``.

``uv add``, ``uv run``, ``uv sync``, and ``uv lock`` are known as "project APIs", and are slightly higher level.
These commands interact with (and require) ``pyproject.toml``, and ``uv`` will ensure your environment is in-sync when they are called,
including creating or updating a `lockfile <https://docs.astral.sh/uv/concepts/projects/sync/>`_,
a universal resolution that is `portable across platforms <https://docs.astral.sh/uv/concepts/resolution/#universal-resolution>`_.

When developing for ``janus-core``, it is usually recommended to use project commands, as described in `Getting started`_
rather than using ``uv pip install`` to modify the project environment manually.

.. tip::

    ``uv`` will detect and use Python versions available on your system,
    but can also be used to `install Python automtically <https://docs.astral.sh/uv/guides/install-python/>`_.
    The desired Python version can be specified when running project commands with the ``--python``/``-p`` option.


For further information, please refer to the `documentation <https://docs.astral.sh/uv/>`_


Running unit tests
++++++++++++++++++

Packages in the ``dev`` dependency group allow tests to be run locally using ``pytest``, by running::

    pytest -v

Alternatively, tests can be run in separate virtual environments using ``tox``::

    tox run -e ALL

This will run all unit tests for multiple versions of Python, in addition to testing that the pre-commit passes, and that documentation builds, mirroring the automated tests on GitHub.

Individual components of the ``tox`` test suite can also be run separately, such as running only running the unit tests with Python 3.9::

    tox run -e py39

See the `tox documentation <https://tox.wiki/>`_ for further options.


Automatic coding style check
++++++++++++++++++++++++++++

Packages in the ``pre-commit`` dependency group allow automatic code formatting and linting on every commit.

To set this up, run::

    pre-commit install

After this, the `ruff linter <https://docs.astral.sh/ruff/linter/>`_, `ruff formatter <https://docs.astral.sh/ruff/formatter/>`_, and `numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`_ (docstring style validator), will run before every commit.

Rules enforced by ruff are currently set up to be comparable to:

- `black <https://black.readthedocs.io>`_ (code formatter)
- `pylint <https://www.pylint.org/>`_ (linter)
- `pyupgrade <https://github.com/asottile/pyupgrade>`_ (syntax upgrader)
- `isort <https://pycqa.github.io/isort/>`_ (import sorter)
- `flake8-bugbear <https://pypi.org/project/flake8-bugbear/>`_ (bug finder)

The full set of `ruff rules <https://docs.astral.sh/ruff/rules/>`_ are specified by the ``[tool.ruff]`` sections of `pyproject.toml <https://github.com/stfc/janus-core/blob/main/pyproject.toml>`_.


Building the documentation
++++++++++++++++++++++++++

Packages in the ``docs`` dependency group install `Sphinx <https://www.sphinx-doc.org>`_ and other packages required to build ``janus-core``'s documentation.

Individual individual documentation pages can be edited directly::

        docs/source/index.rst
        docs/source/user_guide/index.rst
        docs/source/user_guide/get_started.rst
        docs/source/user_guide/tutorial.rst
        docs/source/developer_guide/index.rst
        docs/source/developer_guide/get_started.rst
        docs/source/developer_guide/tutorial.rst
        docs/source/apidoc/janus_core.rst

API documentation is automatically generated from ``docs/source/apidoc/janus_core.rst``.

To document a new module, a new block must be added. For example, for the ``janus_core.calculations.single_point`` module, the following block was added::

    janus\_core.calculations.single\_point module
    ---------------------------------------------

    .. automodule:: janus_core.calculations.single_point
       :members:
       :special-members:
       :private-members:
       :undoc-members:
       :show-inheritance:


``Sphinx`` can then be used to generate the html documentation::

        cd docs
        make clean; make html


Continuous integration
++++++++++++++++++++++

``janus-core`` comes with a ``.github`` folder that contains continuous integration workflows that run on every push and pull request using `GitHub Actions <https://github.com/features/actions>`_. These will:

#. Run all non-optional unit tests
#. Build the documentation
#. Check the coding style conforms by running the pre-commit described above
#. Build and publish tagged commits to PyPI
