"""Set up MLIP preprocessing commandline interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from typer import Option, Typer

from janus_core.cli.types import Tracker

app = Typer()


@app.command()
def preprocess(
    mlip_config: Annotated[
        Path, Option(help="Configuration file to pass to MLIP CLI.", show_default=False)
    ],
    log: Annotated[
        Path, Option(help="Path to save logs to.", rich_help_panel="Logging/summary")
    ] = Path("./janus_results/preprocess-log.yml"),
    tracker: Tracker = True,
    summary: Annotated[
        Path,
        Option(
            help=(
                "Path to save summary of inputs, start/end time, and carbon emissions."
            ),
            rich_help_panel="Logging/summary",
        ),
    ] = Path("./janus_results/preprocess-summary.yml"),
):
    """
    Convert training data to hdf5 by passing a configuration file to the MLIP's CLI.

    Parameters
    ----------
    mlip_config
        Configuration file to pass to MLIP CLI.
    log
        Path to write logs to. Default is Path("preprocess-log.yml").
    tracker
        Whether to save carbon emissions of calculation in log file and summary.
        Default is True.
    summary
        Path to save summary of inputs, start/end time, and carbon emissions. Default
        is Path("preprocess-summary.yml").
    """
    from janus_core.cli.utils import carbon_summary, end_summary, start_summary
    from janus_core.training.preprocess import preprocess as run_preprocess

    config = {
        "mlip_config": mlip_config,
        "log": log,
        "tracker": tracker,
        "summary": summary,
    }

    log_kwargs = {"filemode": "w"}
    if log:
        log_kwargs["filename"] = log

    output_files = {"log": log.absolute()}

    # Save summary information before preprocessing begins
    start_summary(
        command="preprocess",
        summary=summary,
        config=config,
        info={},
        output_files=output_files,
    )

    # Run preprocessing
    run_preprocess(
        mlip_config, attach_logger=True, log_kwargs=log_kwargs, track_carbon=tracker
    )

    # Save carbon summary
    if tracker:
        carbon_summary(summary=summary, log=log)

    # Save time after preprocessing has finished
    end_summary(summary)
