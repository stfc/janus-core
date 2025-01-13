===============
Troubleshooting
===============

Installation
------------

When installing ``janus-core``, CMake errors can occur when building ``phonopy``, such as:

.. collapse:: Example CMake error

    .. code-block:: bash

        CMake Error at /tmp/pip-build-env-j2jx9pp5/overlay/lib/python3.11/site-packages/nanobind/cmake/nanobind-config.cmake:243 (target_compile_features):
            target_compile_features The compiler feature "cxx_std_17" is not known to
            CXX compiler

            "GNU"

            version 4.8.5.
        Call Stack (most recent call first):
            /tmp/pip-build-env-j2jx9pp5/overlay/lib/python3.11/site-packages/nanobind/cmake/nanobind-config.cmake:358 (nanobind_build_library)
            CMakeLists.txt:108 (nanobind_add_module)


        -- Configuring incomplete, errors occurred!

        *** CMake configuration failed
        [end of output]

        note: This error originates from a subprocess, and is likely not a problem with pip.
        ERROR: Failed building wheel for phonopy
        Building wheel for python-hostlist (setup.py) ... done
        Created wheel for python-hostlist: filename=python_hostlist-2.2.1-py3-none-any.whl size=39604 sha256=44f9f27a42895e61a521cf9129a6a3ad03e633b201390da5ef76d5f59db3b94f
        Stored in directory: ...
        Successfully built janus-core python-hostlist
        Failed to build phonopy
        ERROR: ERROR: Failed to build installable wheels for some pyproject.toml based projects (phonopy)


This can typically be resolved by ensuring your C++ compiler is updated, and that the `CXX <https://cmake.org/cmake/help/latest/envvar/CXX.html>`_ environment variable is set.


Carbon tracking
---------------

Enabling tracking (Python)
++++++++++++++++++++++++++

Carbon tracking can be enabled through the ``track_carbon`` option.
By default, this is ``True`` if logging is enabled, but requires setting ``attach_logger``, as this defaults to ``False``.

For example, to track the carbon emissions during a single point calculation:

.. code-block:: python

    from janus_core.calculations.single_point import SinglePoint

    sp = SinglePoint(
        struct_path="tests/data/NaCl.cif",
        attach_logger=True,
        track_carbon=True,
    )

This generates a log file, ``NaCl-singlepoint-log.yml``, which stores the emissions for the calculation.


In the case of multiple calculations, such as geometry optimisation triggered during molecular dynamics,
the emissions for each component of the calculation will be separate items in the log.


Disabling tracking (CLI)
++++++++++++++++++++++++

Currently, carbon tracking is enabled by default when using the command line interface,
saving the total calculating emissions to the generated summary file, as well as additional details and
per-calculation emissions to the log file.

This can be disabled by passing the ``--no-tracker`` flag to any command. For example:

.. code-block:: bash

    janus singlepoint --struct tests/data/NaCl.cif --no-tracker


Sudo access
+++++++++++

On some systems, such as MacOS, the carbon tracker may prompt for your password, if you have sudo access.
To avoid this, you can:

1. Disable carbon tracking, as described in `Disabling tracking (CLI)`_.
3. Modify your sudoers file, as described `here <https://mlco2.github.io/codecarbon/methodology.html#cpu>`_, to provide sudo rights for all future calculations.
2. Provide your password. This may be saved for a period of time, but will need to be entered again in future.
4. Fail authentication, for example by entering an invalid or no password three times, which triggers the tracking to default to a constant power.
