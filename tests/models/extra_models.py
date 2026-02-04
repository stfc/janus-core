"""Download janus_core extra models for tests."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from urllib.request import urlretrieve

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Download extra model files for janus_core tests."
    )
    parser.add_argument("path", type=Path, help="Path to save the downloaded files.")
    args = parser.parse_args()

    urlretrieve(
        "https://zenodo.org/records/16980200/files/NequIP-MP-L-0.1.nequip.zip",
        filename=args.path / "NequIP-MP-L-0.1.nequip.zip",
    )
