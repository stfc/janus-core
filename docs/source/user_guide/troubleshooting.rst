===============
Troubleshooting
===============

Installation
------------

When installing ``janus-core``, CMake errors can occur when building ``phonopy``, often related to the CXX compiler.

Please refer to their `installation documentation <http://phonopy.github.io/phonopy/install.html#missing-or-unknown-cxx-compiler>`_ for guidance.


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
