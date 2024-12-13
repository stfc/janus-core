======================
Continuous Integration
======================

General instructions for setting up self-hosted GitHub runners can be found `here <https://docs.github.com/en/actions/hosting-your-own-runners>`_.

This process typically involves:

1. Add a new self-hosted runner on GitHub
2. Run instructions from GitHub on your runner to download and configure
3. Ensure tags added to the runner are unique, and match those in ``runs-on`` within the CI workflow


MacOS self-hosted runner
========================

Currently, ``janus-core`` uses a self-hosted runner to run all unit tests on MacOS.

To ensure the self-hosted runner remains active, we currently recommend using

.. code-block:: bash

    nohup ./run.sh &!

This runs the script in the background, and disowns the process, allowing the ssh connection to be ended without killing the process.

It would be preferable to `configure the runner application as a service <https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/configuring-the-self-hosted-runner-application-as-a-service?platform=mac>`_,
but currently this appears to lead to difficulties if the GUI is not active
and/or ``sudo`` permissions are not used in running the service.
