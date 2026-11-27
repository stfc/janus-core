===================
MCP server tutorial
===================

This tutorial walks through installing, using, and deploying the ``janus-core`` Model Context Protocol (MCP)
server so AI assistants and custom MCP clients can run atomistic MLIP simulations across all 10
``janus-core`` modes.

For the full reference of CLI options, tools, and resources, see the :doc:`MCP Server User Guide </user_guide/mcp>`.

Prerequisites
=============

Install ``janus-core`` with both the ``mcp`` extra and at least one MLIP extra (such as ``mace``):

.. code-block:: bash

    uv pip install "janus-core[mcp,mace]"

Verify the CLI subcommand is available:

.. code-block:: bash

    janus mcp --help

You should see options for ``--transport`` (``stdio``, ``sse``, ``streamable-http``), ``--host``, ``--port``,
``--mount-path``, and ``--mode``.

Part 1: Inspecting structures and running modes
===============================================

Before launching a network server, you can test the MCP tool functions directly in Python.
First, create a sample NaCl crystal structure:

.. code-block:: python

    from ase.build import bulk
    from ase.io import write

    nacl = bulk("NaCl", "rocksalt", a=5.64)
    write("NaCl.cif", nacl)

Discover available modes and architectures
------------------------------------------

.. code-block:: python

    from janus_core.mcp import inspect_structure, list_architectures, list_modes

    print(list_modes())
    print(list_architectures()["architectures"])
    print(inspect_structure("NaCl.cif"))

``inspect_structure`` reports the chemical formula, atom count, elements, periodic boundary conditions,
lattice lengths/angles, and unit cell volume—useful for validating inputs before running expensive simulations.

Run calculations using dedicated mode tools or ``run_mode``
-----------------------------------------------------------

Every ``janus-core`` mode can be invoked via its dedicated function or via the unified ``run_mode`` dispatcher:

.. code-block:: python

    from janus_core.mcp import run_geomopt, run_mode, run_singlepoint

    # 1. Single-point calculation
    sp = run_singlepoint(
        struct="NaCl.cif",
        arch="mace_mp",
        device="cpu",
        properties=["energy", "forces", "stress"],
    )
    print("Energy (eV):", sp["results"]["energy"])

    # 2. Geometry optimization (uses FrechetCellFilter by default)
    opt = run_geomopt(
        struct="NaCl.cif",
        arch="mace_mp",
        device="cpu",
        fmax=0.05,
        write_results=True,
    )
    print("Converged:", opt["converged"], "Final energy (eV):", opt["energy_ev"])

    # 3. Unified dispatcher (works for all 10 modes)
    eos = run_mode(
        mode="eos",
        params={
            "struct": "NaCl.cif",
            "arch": "mace_mp",
            "device": "cpu",
            "n_volumes": 5,
            "extra_kwargs": {"write_results": False},
        },
    )
    print("Bulk modulus (GPa):", eos["bulk_modulus_gpa"])

Part 2: Connecting a local MCP client (``stdio``)
=================================================

In ``stdio`` mode, the client spawns ``janus-mcp`` as a subprocess and communicates over standard input/output.

Using an AI IDE or desktop assistant
------------------------------------

Add the following entry to your MCP client configuration (Antigravity, Claude Desktop, Cursor, or VS Code):

.. code-block:: json

    {
      "mcpServers": {
        "janus-core": {
          "command": "janus-mcp",
          "args": ["--transport", "stdio"]
        }
      }
    }

Once connected, you can prompt the assistant naturally, for example:

- *"Inspect ``NaCl.cif`` and run a single-point calculation with ``mace_mp`` on CPU."*
- *"Optimize ``NaCl.cif`` with ``FrechetCellFilter`` to ``fmax=0.05`` eV/Å, then compute its equation of state."*
- *"Run a 50-step NVT molecular dynamics simulation of ``NaCl.cif`` at 300 K."*

Calling the server from a Python MCP client
-------------------------------------------

You can also connect programmatically using the official ``mcp`` Python client:

.. code-block:: python

    import asyncio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client


    async def main() -> None:
        server_params = StdioServerParameters(
            command="janus-mcp",
            args=["--transport", "stdio"],
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # List available tools
                tools = await session.list_tools()
                print("Tools:", [tool.name for tool in tools.tools])

                # Inspect a structure via MCP
                info = await session.call_tool(
                    "inspect_structure",
                    arguments={"struct": "NaCl.cif"},
                )
                print("Structure info:", info.content)

                # Run singlepoint via MCP
                sp = await session.call_tool(
                    "singlepoint",
                    arguments={
                        "struct": "NaCl.cif",
                        "arch": "mace_mp",
                        "device": "cpu",
                        "properties": ["energy"],
                    },
                )
                print("Singlepoint result:", sp.content)


    asyncio.run(main())

Part 3: Deploying as a network service
======================================

For remote GPU servers, shared research infrastructure, or containerized workflows, run the server with
``streamable-http`` (recommended) or ``sse`` transport.

Step 1: Launch with Streamable HTTP or SSE
------------------------------------------

Start a Streamable HTTP server exposing only inference modes (omitting ``train`` and ``preprocess``):

.. code-block:: bash

    janus mcp \
      --transport streamable-http \
      --host 127.0.0.1 \
      --port 8000 \
      --mode singlepoint \
      --mode geomopt \
      --mode md \
      --mode eos \
      --mode phonons \
      --mode elasticity \
      --mode neb \
      --mode descriptors

For clients that require Server-Sent Events (SSE):

.. code-block:: bash

    janus mcp --transport sse --host 127.0.0.1 --port 8000

Step 2: Access securely from a remote workstation
-------------------------------------------------

If the server is running on a remote GPU host bound to ``127.0.0.1:8000``, open an SSH tunnel from your laptop:

.. code-block:: bash

    ssh -N -L 8000:127.0.0.1:8000 user@remote-gpu-host

Then configure your local MCP client to connect to ``http://127.0.0.1:8000/mcp`` (Streamable HTTP) or
``http://127.0.0.1:8000/sse`` (SSE).

Step 3: Deploy with Docker Compose
----------------------------------

For reproducible containerized deployments with a persistent workspace directory:

.. code-block:: yaml

    # docker-compose.yml
    services:
      janus-mcp:
        build: .
        image: janus-core-mcp:latest
        ports:
          - "127.0.0.1:8000:8000"
        volumes:
          - ./structures:/workspace/structures
          - ./janus_results:/workspace/janus_results
        command:
          - "--transport"
          - "streamable-http"
          - "--host"
          - "0.0.0.0"
          - "--port"
          - "8000"
        restart: unless-stopped

Launch the container:

.. code-block:: bash

    docker compose up -d

Step 4: Expose over HTTPS with a reverse proxy
----------------------------------------------

When sharing the MCP server across a team network, terminate TLS and enforce authentication in front of
``127.0.0.1:8000``. Example Nginx location block for Streamable HTTP / SSE (with proxy buffering disabled
for streaming responses):

.. code-block:: nginx

    location /mcp {
        proxy_pass http://127.0.0.1:8000/mcp;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_buffering off;
        proxy_read_timeout 3600s;
    }

Next steps
==========

- Explore the full API documentation in :doc:`/apidoc/janus_core` (``janus_core.mcp`` and ``janus_core.cli.mcp``).
- Combine ``janus mcp`` with the calculation tutorials in :doc:`/tutorials/cli/index` and :doc:`/tutorials/python/index`.
