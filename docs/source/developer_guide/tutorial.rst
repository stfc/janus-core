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
        from alignn.ff.ff import (
            AlignnAtomwiseCalculator,
            default_path,
            get_figshare_model_ff,
        )

        # Set default path to directory containing config and model location
        if isinstance(model_path, Path):
            path = model_path
            if path.is_file():
                path = path.parent
        # If a string, assume referring to model_name e.g. "v5.27.2024"
        elif isinstance(model_path, str):
            path = get_figshare_model_ff(model_name=model_path)
        else:
            path = default_path()

        calculator = AlignnAtomwiseCalculator(path=path, device=device, **kwargs)

Most options are unique to the MLIP calculator, so are passed through ``kwargs``.

However, ``device``, referring to the device on which inference should be run, and ``model_path``, referring to the path of an MLIP model, are parameters of ``choose_calculator``, so must be handled explicitly.

In this case, ``device`` is also a parameter of ``AlignnAtomwiseCalculator``, so can be passed through immediately.

``model_path`` requires more care, as ``AlignnAtomwiseCalculator`` does not have a directly corresponding parameter, with ``path`` instead referring to the directory containing the model and configuration files.

Converting ``model_path`` into ``path`` is a minimum requirement, but we also aim to facilitate options native to the MLIP, including a default model, where possible:

- If ``model_path`` refers to the path to the model file, as is expected by ``choose_calculator``, we define ``path`` as the parent directory of ``model_path``
- If ``model_path`` refers to the directory of the model, closer to the typical use of ``AlignnAtomwiseCalculator``, we define ``path`` as ``model_path``
- If ``model_path`` refers to a model label, similar to the MACE ``"small"`` models, we try loading the model using ALIGNN's ``get_figshare_model_ff``
- If ``model_path`` is ``None``, we use the ALIGNN's ``default_path``

.. note::
    ``model_path`` will already be a ``pathlib.Path`` object, if the path exists.

To ensure that the calculator does not receive multiple versions of keywords, it's also necessary to set ``model_path = path``, and remove ``path`` from ``kwargs``.

If the keyword is used by other calculators, this should be done within the ``elif`` branch, but in most cases it can be done automatically by appending ``model_path_kwargs`` within ``_set_model_path``::

    model_path_kwargs = ("model", "model_paths", "potential", "path")

In addition to setting the calculator, ``__version__`` must also imported here, providing a check on the package independent of the calculator itself.

.. note::
    Unlike in other ``janus-core`` modules, any imports required should be contained within the ``elif`` branch, as these dependencies are optional.


4. Add tests
------------

Tests must be added to ensure that, at a minimum, the new calculator allows an MLIP to be loaded correctly, and that an energy can be calculated.

This can be done by adding the appropriate data as tuples to the ``pytest.mark.parametrize`` lists in the ``tests.test_mlip_calculators`` and ``tests.test_single_point`` modules.

For ``tests.test_mlip_calculators``, ``architecture``, ``device`` and accepted forms of ``model_path`` should be tested, ensuring that the calculator and its version are correctly set::

    @pytest.mark.extra_mlips
    @pytest.mark.parametrize(
        "architecture, device, kwargs",
        [
            ("alignn", "cpu", {}),
            ("alignn", "cpu", {"model_path": "tests/models/v5.27.2024"}),
            ("alignn", "cpu", {"model_path": "tests/models/v5.27.2024/best_model.pt"}),
            ("alignn", "cpu", {"model": "alignnff_wt10"}),
            ("alignn", "cpu", {"path": "tests/models/v5.27.2024"}),
        ],
    )
    def test_extra_mlips(architecture, device, kwargs):

It is also useful to test that ``model_path``, and ``model`` or and the "standard" MLIP calculator parameter (``path``) cannot be defined simultaneously::

    @pytest.mark.extra_mlips
    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "model_path": "tests/models/v5.27.2024/best_model.pt",
                "model": "tests/models/v5.27.2024/best_model.pt",
            },
            {
                "model_path": "tests/models/v5.27.2024/best_model.pt",
                "path": "tests/models/v5.27.2024/best_model.pt",
            },
        ],
    )
    def test_extra_mlips_invalid(kwargs):

For ``tests.test_single_point``, ``architecture``, ``device``, and the potential energy of NaCl predicted by the MLIP should be defined, ensuring that calculations can be performed::

    test_extra_mlips_data = [("alignn", "cpu", -11.148092269897461)]

Running these tests requires an additional flag to be passed to ``pytest``::

    pytest -v --run-extra-mlips

Alternatively, using ``tox``::

    tox -e extra-mlips

Adding a new Observable
=======================

Additional built-in observable quantities may be added for use by the ``janus_core.helpers.correlator.Correlation`` class. These should conform to the ``__call__`` signature of ``janus_core.helpers.janus_types.Observable``. For a user this can be accomplished by writing a function, or class also implementing a commensurate ``__call__``.

Built-in observables are collected within ``janus_core.helpers.observables``. For example the ``janus_core.helpers.observables.Stress`` observable allows a user to quickly setup a given correlation of stress tensor components (with and without the ideal gas constribution). An observable for the ``xy`` component is obtained without the ideal gas contribution as::

    Stress("xy", False)

A new built-in observables can be implemented by a class with the method::

   def __call__(self, atoms: Atoms, *args, **kwargs) -> float

The ``__call__`` should contain all the logic for obtaining some ``float`` value from an ``Atoms`` object, alongside optional positional arguments and kwargs. The args and kwargs are set by a user when specifying correlations for a ``janus_core.calculations.md.MolecularDynamics`` run. See also ``janus_core.helpers.janus_types.CorrelationKwargs``. These are set at the instantiation of the ``janus_core.calculations.md.MolecularDynamics`` object and are not modified. These could be used e.g. to specify an observable calculated only from one atom's data.

``janus_core.helpers.observables.Stress`` includes a constructor to take a symbolic component, e.g. ``"xx"`` or ``"yz"``, and determine the index required from ``ase.Atoms.get_stress`` on instantiation for ease of use.
