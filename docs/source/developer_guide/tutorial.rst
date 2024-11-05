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

New MLIPs should be added as optional dependencies under ``[tool.poetry.dependencies]``, and added as an ``extra`` under ``[tool.poetry.extras]``::

    [tool.poetry.dependencies]
    alignn = { version = "2024.5.27", optional = true }
    sevenn = { version = "0.10.0", optional = true }
    torch_geometric = { version = "^2.5.3", optional = true }

    [tool.poetry.extras]
    alignn = ["alignn"]
    sevennet = ["sevenn", "torch_geometric"]

Poetry will automatically resolve dependencies of the MLIP, if present, to ensure consistency with existing dependencies.

.. note::
    In the case of ``sevennet``, it was necessary to add ``torch_geometric`` as an additional dependency, but in most cases this should not be required

Extra dependencies can then be installed by running:

.. code-block:: bash

    poetry lock
    poetry install --extras "alignn sevennet"

or, for all extras:

.. code-block:: bash

    poetry install --extras all


2. Register MLIP architecture
-----------------------------

In order to be able to select the appropriate MLIP when running calculations, it is first necessary to add a label for the architecture to ``Architectures`` in ``janus_core.helpers.janus_types``.

In this case, we choose the label ``"alignn"``::

    Architectures = Literal["mace", "mace_mp", "mace_off", "m3gnet", "chgnet", "alignn"]


3. Add MLIP calculator
----------------------

Next, we need to allow the ASE calculator corresponding to the MLIP label to be set.

This is done within the ``janus_core.helpers.mlip_calculators`` module, if ``arch`` matches the label defined above::

    elif arch == "alignn":
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
    Some MLIPs do not support this, so you may be required to cast it back to a string (``str(model_path)``).

To ensure that the calculator does not receive multiple versions of keywords, it's also necessary to set ``model_path = path``, and remove ``path`` from ``kwargs``.

If the keyword is used by other calculators, this should be done within the ``elif`` branch, but in most cases it can be done automatically by appending ``model_path_kwargs`` within ``_set_model_path``::

    model_path_kwargs = ("model", "model_paths", "potential", "path")

In addition to setting the calculator, ``__version__`` must also imported here, providing a check on the package independent of the calculator itself.

.. note::
    Unlike in other ``janus-core`` modules, any imports required should be contained within the ``elif`` branch, as these dependencies are optional.


4. Add tests
------------

Tests must be added to ensure that, at a minimum, the new calculator allows an MLIP to be loaded correctly, and that an energy can be calculated.

This can be done by adding the appropriate data as tuples to the ``pytest.mark.parametrize`` lists in the ``tests.test_mlip_calculators`` and ``tests.test_single_point`` modules
that reside in files ``tests/test_mlip_calculators.py``` and ``tests/test_single_point.py``, respectively.


Load models - success
^^^^^^^^^^^^^^^^^^^^^

For ``tests.test_mlip_calculators``, ``arch``, ``device`` and accepted forms of ``model_path`` should be tested, ensuring that the calculator and its version are correctly set::

    @pytest.mark.extra_mlips
    @pytest.mark.parametrize(
        "arch, device, kwargs",
        [
            ("alignn", "cpu", {}),
            ("alignn", "cpu", {"model_path": "tests/models/v5.27.2024"}),
            ("alignn", "cpu", {"model_path": "tests/models/v5.27.2024/best_model.pt"}),
            ("alignn", "cpu", {"model": "alignnff_wt10"}),
            ("alignn", "cpu", {"path": "tests/models/v5.27.2024"}),
        ],
    )
    def test_extra_mlips(arch, device, kwargs):

.. note::
    Not all models support an empty (default) model path, so the equivalent test to``("alignn", "cpu", {})`` may need to be removed, or moved to the tests described in `Load models - failure`_.

Load models - failure
^^^^^^^^^^^^^^^^^^^^^

It is also useful to test that ``model_path``, and ``model`` or and the "standard" MLIP calculator parameter (``path``) cannot be defined simultaneously

.. code-block:: python

    @pytest.mark.extra_mlips
    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "arch": "alignn",
                "model_path": MODEL_PATH / "v5.27.2024" / "best_model.pt",
                "model": MODEL_PATH / "v5.27.2024" / "best_model.pt",
            },
            {
                "arch": "alignn",
                "model_path": "tests/models/v5.27.2024/best_model.pt",
                "path": "tests/models/v5.27.2024/best_model.pt",
            },
        ],
    )
    def test_extra_mlips_invalid(kwargs):

Test correctness
^^^^^^^^^^^^^^^^

For ``tests.test_single_point``, ``arch``, ``device``, and the potential energy of NaCl predicted by the MLIP should be defined, ensuring that calculations can be performed::

    test_extra_mlips_data = [("alignn", "cpu", -11.148092269897461, {})]


Running these tests requires an additional flag to be passed to ``pytest``::

    pytest -v --run-extra-mlips

Alternatively, using ``tox``::

    tox -e extra-mlips

Adding a new Observable
=======================

Additional built-in observable quantities may be added for use by the ``janus_core.processing.correlator.Correlation`` class. These should extend ``janus_core.processing.observables.Observable``. The abstract method ``__call__`` should be implemented to obtain the values of the observed quantity from an ``Atoms`` object:

.. code-block:: python

    @abstractmethod
    def __call__(self, atoms: Atoms) -> list[float]

These values are returned as a ``list[float]`` representing the dimensions of the observed value which may be returned for a slice of atoms individually. ``value_count`` should be overloaded to return the expected length returned by ``__call__``. This is so enough correlators may be spawned to track the correlated the values. Note the final correlation will be the average of the correlations across these dimensions and atom counts.

For example the ``janus_core.processing.observables.Stress`` built-in returns the required stress components as calculated by ``Atoms.get_stress`` called upon the tracked atoms. Therefore ``__call__`` returns one ``float`` per component correlated. For example an observable representing the ``xy`` and ``zy`` components of stress computed across all odd-index atoms in some ``Atoms`` object can be constructed as follows:

.. code-block:: python

    s = Stress(components=["xy", "zy"], atoms_slice=(0, None, 2))

In this case ``s(atoms)`` will return 2 values.

The ``janus_core.processing.observables.Velocity`` built-in's ``__call__`` not only returns atom velocity for the requested dimensions, but also returns them for every tracked atom. ``value_count`` is overloaded to reflect this. Therefore given the observable

.. code-block:: python

    v = Velocity(components=["x", "y", "z"], atoms_slice=(0, None, 2))

The value of ``v(atoms)`` will have the length ``3 * len(atoms)//2``. The 3 dimensions for each odd-indexed atom.

New built-in observables are collected within the ``janus_core.processing.observables`` module. Special cases may also be defined for ease of use:

.. code-block:: python

    StressHydrostatic = Stress(components=["xx", "yy", "zz"])
    StressShear = Stress(components=["xy", "yz", "zx"])
