========
Tutorial
========

Adding a new MLIP
=================

Integration of new MLIPs currently requires a corresponding `ASE calculator <https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html>`_.

The following steps can then be taken, using `ALIGNN-FF <https://github.com/usnistgov/alignn>`_ as an example:

1. Add required dependencies
----------------------------

Dependencies for ``janus-core`` are specified through a ``pyproject.toml`` file, with syntax defined by `poetry's dependency specification <https://python-poetry.org/docs/dependency-specification/>`_.

Required dependencies are listed under ``[tool.poetry.dependencies]``, but new MLIPs should initially be added as optional dependencies under ``[tool.poetry.group.extra-mlips.dependencies]``::

    [tool.poetry.group.extra-mlips]
    optional = true
    [tool.poetry.group.extra-mlips.dependencies]
    alignn = "^2024.5.27"

Poetry will automatically resolve dependencies of the MLIP, if present, to ensure consistency with existing dependencies.

These dependencies can then be installed by running:

.. code-block:: bash

    poetry lock
    poetry install --with extra-mlips


2. Register MLIP architecture
-----------------------------

In order to be able to select the appropriate MLIP when running calculations, it is first necessary to add a label for the architecture to ``Architectures`` in ``janus_core.helpers.janus_types``.

In this case, we choose the label ``"alignn"``::

    Architectures = Literal["mace", "mace_mp", "mace_off", "m3gnet", "chgnet", "alignn"]


3. Add MLIP calculator
----------------------

Next, we need to allow the ASE calculator corresponding to the MLIP label to be set.

This is done within the ``janus_core.helpers.mlip_calculators`` module, if ``architecture`` matches the label defined above::

    elif architecture == "alignn":
        from alignn import __version__
        from alignn.ff.ff import AlignnAtomwiseCalculator, default_path

        # Set default path for config and model location
        kwargs.setdefault("path", default_path())
        calculator = AlignnAtomwiseCalculator(device=device, **kwargs)

Note, unlike in other ``janus-core`` modules, any imports required should be contained within the ``elif`` branch, as these dependencies are optional.

As ``device`` is a parameter of ``choose_calculator``, it must either be passed explicitly, or added to the ``kwargs``, with the appropriate keyword expected by the calculator. When using CHGNet, for example, we instead set ``use_device=device``.

All other parameters are passed through ``kwargs`` specific to the calculator, although extra handling of common parameters, such as the model (path), may be useful.

In this case, we define a default ``"path"``, setting both the model and configuration, unless either is explicitly overridden.

In addition to setting the calculator, ``__version__`` must also imported here, providing a check on the package independent of the calculator itself.


4. Add tests
------------

Tests must be added to ensure that, at a minimum, the new calculator allows an MLIP to be loaded correctly, and that an energy can be calculated.

This can be done by adding the appropriate data as tuples to the ``test_extra_mlips_data`` lists in the ``tests.test_mlip_calculators`` and ``tests.test_single_point`` modules.

For ``tests.test_mlip_calculators``, ``architecture`` and ``device`` should be defined, ensuring that the calculator and its version are correctly set::

    test_extra_mlips_data = [("alignn", "cpu")]

For ``tests.test_single_point``, ``architecture``, ``device``, and the predicted potential energy of NaCl should be defined, ensuring that calculations can be performed::

    test_extra_mlips_data = [("alignn", "cpu", -11.148092269897461)]

Running these tests requires an additional flag to be passed to ``pytest``::

    pytest -v --run-extra-mlips

Alternatively, using ``tox``::

    tox -e extra-mlips
