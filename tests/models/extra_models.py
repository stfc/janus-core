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

    args.path.mkdir(parents=True, exist_ok=True)

    urlretrieve(
        "https://zenodo.org/records/16980200/files/NequIP-MP-L-0.1.nequip.zip",
        filename=args.path / "NequIP-MP-L-0.1.nequip.zip",
    )

    urlretrieve(
        "https://github.com/MDIL-SNU/SevenNet/raw/dff008ac9c53d368b5bee30a27fa4bdfd73f19b2/sevenn/pretrained_potentials/SevenNet_l3i5/checkpoint_l3i5.pth",
        filename=args.path / "SevenNet_l3i5.pth",
    )
