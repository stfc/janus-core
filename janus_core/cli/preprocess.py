# noqa: I002, FA102
"""Set up MLIP preprocessing commandline interface."""

# Issues with future annotations and typer
# c.f. https://github.com/maxb2/typer-config/issues/295
# from __future__ import annotations

from pathlib import Path
from typing import Annotated

from typer import Option, Typer

app = Typer()


@app.command()
def preprocess(
    mlip_config: Annotated[
        Path, Option(help="Configuration file to pass to MLIP CLI.")
    ],
    log: Annotated[Path, Option(help="Path to save logs to.")] = Path(
        "preprocess-log.yml"
    ),
    tracker: Annotated[
        bool, Option(help="Whether to save carbon emissions of calculation")
    ] = True,
    summary: Annotated[
        Path,
        Option(
            help=(
                "Path to save summary of inputs, start/end time, and carbon emissions."
            )
        ),
    ] = Path("preprocess-summary.yml"),
):
    """
    Convert training data to hdf5 by passing a configuration file to the MLIP's CLI.

    Parameters
    ----------
    mlip_config : Path
        Configuration file to pass to MLIP CLI.
    log : Optional[Path]
        Path to write logs to. Default is Path("preprocess-log.yml").
    tracker : bool
        Whether to save carbon emissions of calculation in log file and summary.
        Default is True.
    summary : Optional[Path]
        Path to save summary of inputs, start/end time, and carbon emissions. Default
        is Path("preprocess-summary.yml").
    """
    from janus_core.cli.utils import carbon_summary, end_summary, start_summary
    from janus_core.training.preprocess import preprocess as run_preprocess

    inputs = {"mlip_config": str(mlip_config)}

    # Save summary information before preprocessing begins
    start_summary(command="preprocess", summary=summary, inputs=inputs)

    log_kwargs = {"filemode": "w"}
    if log:
        log_kwargs["filename"] = log

    # Run preprocessing
    run_preprocess(
        mlip_config, attach_logger=True, log_kwargs=log_kwargs, track_carbon=tracker
    )

    # Save carbon summary
    if tracker:
        carbon_summary(summary=summary, log=log)

    # Save time after preprocessing has finished
    end_summary(summary)
