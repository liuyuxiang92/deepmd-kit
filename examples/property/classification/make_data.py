# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate a toy binary H2 bond-length dataset, not physical phase labels."""

from pathlib import (
    Path,
)

import numpy as np


def main() -> None:
    """Write deterministic training and validation systems next to this script."""
    rng = np.random.default_rng(42)
    for split, nframes in (("train", 32), ("valid", 16)):
        system = Path(__file__).parent / "data" / split
        data = system / "set.000"
        data.mkdir(parents=True, exist_ok=True)
        labels = np.arange(nframes) % 2
        coords = np.zeros((nframes, 2, 3))
        coords[:, 1, 0] = 0.75 + labels * 0.75 + rng.normal(0, 0.025, nframes)
        np.save(data / "coord.npy", coords.reshape(nframes, -1))
        np.save(data / "class.npy", labels.reshape(-1, 1))
        (system / "type.raw").write_text("0 0\n")
        (system / "type_map.raw").write_text("H\n")
        (system / "nopbc").touch()


if __name__ == "__main__":
    main()
