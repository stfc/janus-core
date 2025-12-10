========
Tutorial
========

Adding a new MLIP
=================

Integration of new MLIPs currently requires a corresponding `ASE calculator <https://ase-lib.org/ase/calculators/calculators.html>`_.

The following steps can then be taken, using `ORB <https://github.com/orbital-materials/orb-models>`_ as an example:

1. Add required dependencies
----------------------------

Dependencies for ``janus-core`` are specified through a ``pyproject.toml`` file,
with syntax consistent with `PEP 621 <https://docs.astral.sh/uv/concepts/projects/dependencies/#project-dependencies>`_.

New MLIPs should be added as optional dependencies under ``[project.optional-dependencies]``,
both with its own label, and to ``all`` if it is compatible with existing MLIPs::

    [project.optional-dependencies]
    chgnet = [
        "chgnet == 0.4.0",
    ]
    mace = [
        "mace-torch==0.3.10",
        "torch-dftd==0.4.0",
    ]

    orb = [
        "orb-models == 0.4.2",
        "pynanoflann",
    ]

    all = [
        "janus-core[chgnet]",
        "janus-core[mace]",
        "janus-core[orb]",
    ]

    # MLIPs with dgl dependency
    alignn = [
        "alignn == 2024.5.27",
        "torch == 2.2",
        "torchdata == 0.7.1",
    ]
    m3gnet = [
        "matgl == 1.1.3",
        "torch == 2.2",
        "torchdata == 0.7.1",
    ]

    [tool.uv.sources]
    pynanoflann = { git = "https://github.com/dwastberg/pynanoflann", rev = "af434039ae14bedcbb838a7808924d6689274168" }


``uv`` will automatically resolve dependencies of the MLIP, if present, to ensure consistency with existing dependencies.

.. note::

    In most cases, sub-dependencies should automatically be included with the specified MLIP, but a few exceptions can be seen.

    For example, ``orb`` requires ``pynanoflann``, but does not list it as a dependency. ``pynanoflann`` requires a specific git commit,
    defined in ``[tool.uv.sources]``. Similarly, ``torch-dftd`` is not listed as a dependency of ``mace``, so has been manually included.

    In the case of ``alignn`` and ``m3gnet``, ``torch`` and ``torchdata`` are listed dependencies, but are pinned to ensure compatibility.


Extra dependencies can then be installed by running:

.. code-block:: bash

    uv sync --extra mace --extra orb


or, for all compatible extras:

.. code-block:: bash

    uv sync --extra all


If a new MLIP is not compatible with others, this must be declared as a conflict. For example:

.. code-block:: bash

    conflicts = [
        [
            { extra = "chgnet" },
            { extra = "alignn" },
        ],
        [
            { extra = "chgnet" },
            { extra = "m3gnet" },
        ],
        [
            { extra = "all" },
            { extra = "alignn" },
        ],
        [
            { extra = "all" },
            { extra = "m3gnet" },
        ],
    ]


This states that ``m3gnet`` and ``alignn`` both conflict with ``chgnet``, and by extension, ``all`` (due to different ``torch`` requirements).

2. Register MLIP architecture
-----------------------------

In order to be able to select the appropriate MLIP when running calculations, it is first necessary to add a label for the architecture to ``Architectures`` in ``janus_core.helpers.janus_types``.

In this case, we choose the label ``"orb"``:

.. code-block:: python

    Architectures = Literal[
        "mace",
        "mace_mp",
        "mace_off",
        "m3gnet",
        "chgnet",
        "alignn",
        "orb",
    ]


3. Add MLIP calculator
----------------------

Next, we need to allow the ASE calculator corresponding to the MLIP label to be set.

This is done within the ``janus_core.helpers.mlip_calculators`` module, if ``arch`` matches the label defined above:

.. code-block:: python

    elif arch == "orb":
        from orb_models import __version__
        from orb_models.forcefield.calculator import ORBCalculator
        from orb_models.forcefield.direct_regressor import DirectForcefieldRegressor
        import orb_models.forcefield.pretrained as orb_ff

        # Default model
        model = model if model else "orb_v3_conservative_20_omat"

        if isinstance(model, DirectForcefieldRegressor):
            loaded_model = model
            model = "loaded_DirectForcefieldRegressor"
        else:
            try:
                loaded_model = getattr(orb_ff, model.replace("-", "_"))()
            except AttributeError as e:
                raise ValueError(
                    "`model` must be a `DirectForcefieldRegressor`, pre-trained "
                    "model label (e.g. 'orb-v2'), or `None` (uses default, orb-v2)"
                ) from e

        calculator = ORBCalculator(model=loaded_model, device=device, **kwargs)


Most options are unique to the MLIP calculator, so are passed through ``kwargs``.

However, ``device``, referring to the device on which inference should be run, and ``model``, referring to the name or path of an MLIP model, are parameters of ``choose_calculator``, so must be handled explicitly.

In this case, ``device`` is also a parameter of ``ORBCalculator``, so can be passed through immediately.

``model`` requires more care, as ``ORBCalculator`` does not have a directly corresponding parameter, with ``model`` instead referring to the loaded ``GraphRegressor`` model.

Converting ``model`` into the form expected by ``ORBCalculator`` is a minimum requirement,
but we also aim to facilitate options native to the MLIP, including a default model, where possible:

- If ``model`` is ``None``, we use a default ORB model, ``orb_v2``
- If ``model`` is a loaded ``GraphRegressor``, we pass it through to ``ORBCalculator``, while renaming ``model`` for labelling purposes
- If ``model`` refers to a model label, similar to the MACE ``"small"`` models, we try loading the model using ORB's ``orb_ff`` load functions

.. note::

    ``model`` will already be a ``pathlib.Path`` object, if the path exists.
    Some MLIPs do not support this, so you may be required to cast it back to a string (``str(model)``).


To ensure that the calculator does not receive multiple versions of keywords, it's also necessary to set ``model = path``, and remove ``path`` from ``kwargs``.

If the keyword is used by other calculators, this should be done within the ``elif`` branch, but in most cases it can be done automatically by appending ``model_kwargs`` within ``_set_model``::

    model_kwargs = {"model_paths", "potential", "path"}

In addition to setting the calculator, ``__version__`` must also imported here, providing a check on the package independent of the calculator itself.

.. note::

    Unlike in other ``janus-core`` modules, any imports required should be contained within the ``elif`` branch, as these dependencies are optional.


4. Add tests
------------

Tests must be added to ensure that, at a minimum, the new calculator allows an MLIP to be loaded correctly, and that an energy can be calculated.

This can be done by adding the appropriate data as tuples to the ``pytest.mark.parametrize`` lists in the ``tests.test_mlip_calculators`` and ``tests.test_single_point`` modules
that reside in files ``tests/test_mlip_calculators.py`` and ``tests/test_single_point.py``, respectively.


Skip tests
^^^^^^^^^^

To allow arbitrary combinations of MLIPs to be installed, tests for non-MACE extras are designed to be skipped unless their module can be imported.

This is done using the ``skip_extras`` function, defined in ``tests/utils.py``. This should be modified for new extras:

.. code-block:: python

    def skip_extras(arch: str):
        match arch:
            case "orb":
                pytest.importorskip("orb_models")


Load models - success
^^^^^^^^^^^^^^^^^^^^^

For ``tests.test_mlip_calculators``, ``arch``, ``device`` and accepted forms of ``model`` should be tested, ensuring that the calculator and its version are correctly set:

.. code-block:: python

    @pytest.mark.parametrize(
        "arch, device, kwargs",
        [
            ("orb", "cpu", {}),
            ("orb", "cpu", {"model": ORB_MODEL}),
        ],
    )
    def test_mlips(arch, device, kwargs):


.. note::

    Not all models support an empty (default) model path, so the equivalent test to ``("orb", "cpu", {})`` may need to be removed,
    or moved to the tests described in `Load models - failure`_.


Load models - failure
^^^^^^^^^^^^^^^^^^^^^

It is also useful to test that invalid model paths are handled as expected:

.. code-block:: python

    @pytest.mark.parametrize(
        "arch, model",
        [
            ("orb", "/invalid/path"),
        ],
    )
    def test_invalid_model(arch, model):


and that ``model`` and "standard" MLIP calculator parameter (``path``) cannot be defined simultaneously:

.. code-block:: python

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"arch": "mace", "model": MACE_MP_PATH, "model_paths": MACE_MP_PATH},
            {"arch": "orb", "model": ORB_MODEL, "path": ORB_MODEL},
        ],
    )
    def test_duplicate_model_input(kwargs):


Test correctness
^^^^^^^^^^^^^^^^

For ``tests.test_single_point``, ``arch``, ``device``, and the potential energy of a structure predicted by the MLIP should be defined,
ensuring that calculations can be performed:

.. code-block:: python

    @pytest.mark.parametrize(
        "arch, device, expected_energy, struct, kwargs",
        [
            ("orb", "cpu", -27.088973999023438, "NaCl.cif", {}),
            ("orb", "cpu", -27.088973999023438, "NaCl.cif", {"model": "orb-v2"}),
        ],
    )
    def test_extras(arch, device, expected_energy, struct, kwargs):



Adding a new Observable
=======================

A :class:`janus_core.processing.observables.Observable` abstracts obtaining a quantity derived from ``Atoms``. They may be used as kernels for input into analysis such as a correlation.

Additional built-in observable quantities may be added for use by the :class:`janus_core.processing.correlator.Correlation` class. These should extend :class:`janus_core.processing.observables.Observable` and are implemented within the :py:mod:`janus_core.processing.observables` module.

The abstract method ``__call__`` should be implemented to obtain the values of the observed quantity from an ``Atoms`` object. When used as part of a :class:`janus_core.processing.correlator.Correlation`, each value will be correlated and the results averaged.

As an example of building a new ``Observable`` consider the :class:`janus_core.processing.observables.Stress` built-in. The following steps may be taken:

1. Defining the observable.
---------------------------

The stress tensor may be computed on an atoms object using ``Atoms.get_stress``. A user may wish to obtain a particular component, or perhaps only compute the stress on some subset of ``Atoms``. For example during a :class:`janus_core.calculations.md.MolecularDynamics` run a user may wish to correlate only the off-diagonal components (shear stress), computed across all atoms.

2. Writing the ``__call__`` method.
-----------------------------------

In the call method we can use the base :class:`janus_core.processing.observables.Observable`'s optional atom selector ``atoms_slice`` to first define the subset of atoms to compute the stress for:

.. code-block:: python

    def __call__(self, atoms: Atoms) -> list[float]:
        sliced_atoms = atoms[self.atoms_slice]
        # must be re-attached after slicing for get_stress
        sliced_atoms.calc = atoms.calc

Next the stresses may be obtained from:

.. code-block:: python

    stresses = (
            sliced_atoms.get_stress(
                include_ideal_gas=self.include_ideal_gas, voigt=True
            )
            / units.GPa
        )

Finally, to facilitate handling components in a symbolic way, :class:`janus_core.processing.observables.ComponentMixin` exists to parse ``str`` symbolic components to ``int`` indices by defining a suitable mapping. For the stress tensor (and the format of ``Atoms.get_stress``) a suitable mapping is defined in :class:`janus_core.processing.observables.Stress`'s ``__init__`` method:

.. code-block:: python

        ComponentMixin.__init__(
            self,
            components={
                "xx": 0,
                "yy": 1,
                "zz": 2,
                "yz": 3,
                "zy": 3,
                "xz": 4,
                "zx": 4,
                "xy": 5,
                "yx": 5,
            },
        )

This then concludes the ``__call__`` method for :class:`janus_core.processing.observables.Stress` by using :class:`janus_core.processing.observables.ComponentMixin`'s
pre-calculated indices:

.. code-block:: python

    return stesses[self._indices]

The combination of the above means a user may obtain, say, the ``xy`` and ``zy`` stress tensor components over odd-indexed atoms by calling the following observable on an ``Atoms``:

.. code-block:: python

    s = Stress(components=["xy", "zy"], atoms_slice=(0, None, 2))


Since usually total system stresses are required we can define two built-ins to handle the shear and hydrostatic stresses like so:

.. code-block:: python

    StressHydrostatic = Stress(components=["xx", "yy", "zz"])
    StressShear = Stress(components=["xy", "yz", "zx"])

Where by default :class:`janus_core.processing.observables.Observable`'s ``atoms_slice`` is ``slice(0, None, 1)``, which expands to all atoms in an ``Atoms``.

For comparison the :class:`janus_core.processing.observables.Velocity` built-in's ``__call__`` not only returns atom velocity for the requested components, but also returns them for every tracked atom i.e:

.. code-block:: python

    def __call__(self, atoms: Atoms) -> list[float]:
        return atoms.get_velocities()[self.atoms_slice, :][:, self._indices].flatten()


Deprecating a parameter
=======================

When deprecating a parameter, we aim to include a warning for at least one release cycle
before removing the original parameter, to minimise the impact on users.

This deprecation period will be extended following the release of v1.

Deprecation should be handled for both the CLI and Python interfaces,
as described below for  the requirements in renaming ``model_path`` to ``model``,
as well as renaming ``filter_func`` to ``filter_class`` when these requirements differ.


Python interface
----------------

1. Update the parameters.

In addition to adding the new parameter, the old parameter default should be changed to ``None``:

.. code-block:: python

    def __init__(
        self,
        ...
        model: PathLike | None = None,
        model_path: PathLike | None = None,
        ...
        filter_func: Callable | str | None = None,
        filter_kwargs: dict[str, Any] | None = None,
    )

All references to the old paramter must be updated, which may include cases where separate calculations interact:

.. code-block:: python

    self.minimize_kwargs.setdefault("filter_class", None)


2. Handle the duplicated parameters.

If both are specifed, we should raise an error if possible:

.. code-block:: python

    if model_path:
        # `model` is a new parameter, so there is no reason to be using both
        if model:
            raise ValueError(
                "`model` has replaced `model_path`. Please only use `model`"
            )
        self.model = model_path


If the new parameter's default is not ``None``, it's usually acceptable to allow both,
prefering the old option, as this is only not ``None`` if it has been set explicitly:

.. code-block:: python

        if filter_func:
            filter_class = filter_func


3. Raise a ``FutureWarning`` if the old parameter is set.

This usually needs to be later than the duplicate parameters are handled,
to ensure logging has been set up:

.. code-block:: python

    if model_path:
        warn(
            "`model_path` has been deprecated. Please use `model`.",
            FutureWarning,
            stacklevel=2,
        )


4. Update the parameter docstrings:

.. code-block:: python

    """"
    Parameters
    ----------
    ...
    model
        MLIP model label, path to model, or loaded model. Default is `None`.
    model_path
        Deprecated. Please use `model`.
    ...
    """"


5. Test that the old option still correctly sets the new parameter.

This should also check that a ``FutureWarning`` is raised:

.. code-block:: python

    def test_deprecation_model_path():
        """Test FutureWarning raised for model_path."""
        skip_extras("mace")

        with pytest.warns(FutureWarning, match="`model_path` has been deprecated"):
            sp = SinglePoint(
                arch="mace_mp",
                model_path=MACE_PATH,
                struct=DATA_PATH / "NaCl.cif",
            )

        assert sp.struct.calc.parameters["model"] == str(MACE_PATH.as_posix())


6. Replace the old option in tests and documentation, including:

- README
- Python user guide
- Python tutorial notebooks
- Python tests


Command line
------------

1. Add the new parameter:

.. code-block:: python

    Model = Annotated[
        str | None,
        Option(
            help="MLIP model name, or path to model.", rich_help_panel="MLIP calculator"
        ),
    ]

2. Add the ``deprecated_option`` callback to the old parameter,
which prints a warning if it is used, and hide the option:

.. code-block:: python

    from janus_core.cli.utils import deprecated_option

    ModelPath = Annotated[
        str | None,
        Option(
            help="Deprecated. Please use --model",
            rich_help_panel="MLIP calculator",
            callback=deprecated_option,
            hidden=True,
        ),
    ]

3. Update the docstrings, with a deprecation note for consistency:

.. code-block:: python

    """
    Parameters
    ----------
    ...
    model
        Path to MLIP model or name of model. Default is `None`.
    model_path
        Deprecated. Please use `model`.
    ...
    """"


4. Handle the duplicate values.

If the parameter is passed directly to the Python interface, and no other parameters depend on it,
it may be sufficient to pass both through and handle it within the Python interface, as is the case for ``model_path``:

.. code-block:: python

    singlepoint_kwargs = {
        ...
        "model": model,
        "model_path": model_path,
        ...
    }


However, it may be necessary to ensure only one value has been specified before the Python interface is reached,
as is the case for ``filter_func`` in ``geomopt``:

.. code-block:: python

    if filter_func and filter_class:
        raise ValueError("--filter-func is deprecated, please only use --filter")


5. Test that the old option still correctly sets the new parameter.

Ideally, this would also check that a ``FutureWarning`` is raised,
but capture of this is inconsistent during tests, so we do not currently test against this:

.. code-block:: python

    def test_model_path_deprecated(tmp_path):
        """Test model_path sets model."""
        file_prefix = tmp_path / "NaCl"
        results_path = tmp_path / "NaCl-results.extxyz"
        log_path = tmp_path / "test.log"

        result = runner.invoke(
            app,
            [
                "singlepoint",
                "--struct",
                DATA_PATH / "NaCl.cif",
                "--arch",
                "mace_mp",
                "--model-path",
                MACE_PATH,
                "--log",
                log_path,
                "--file-prefix",
                file_prefix,
                "--no-tracker",
            ],
        )
        assert result.exit_code == 0

        atoms = read(results_path)
        assert "model" in atoms.info
        assert atoms.info["model"] == str(MACE_PATH.as_posix())

6. Replace the old CLI option in tests and documentation, including:

- README
- CLI user guide
- CLI tutorial notebooks
- CLI tests
