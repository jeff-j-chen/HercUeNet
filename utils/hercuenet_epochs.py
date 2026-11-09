"""HercUeNet (the mix28_ps2 run) at depth 0 for every saved epoch checkpoint, on the sheet of utils/prize_depth_grid.py (DEPTH_GRID_SHEET).

row 1: every epoch on its own p1 -> p99.5 scale; row 2: every epoch on epoch 4's scale (a quieter epoch looks quieter)

python3 -m utils.hercuenet_epochs
"""
from __future__ import annotations

import cv2
import numpy as np

from utils import mix6, prize_depth_grid as grid
from utils.fast_figure import colorize, orient_display, save_panel_grid

RUN = grid.CUE_CKPT.parent
EPOCHS = [n for n in range(1, 10) if (RUN / f"epoch_{n}.pth").is_file()]


def main() -> None:
    for epoch in EPOCHS:
        grid.run_hercuenet(RUN / f"epoch_{epoch}.pth", (0,), f"epoch{epoch}")
    mask = grid._mask()
    maps = {epoch: grid.hercuenet_map(0, mask, f"epoch{epoch}") for epoch in EPOCHS}
    own = {epoch: np.percentile(array[np.isfinite(array)], (1.0, grid.STRETCH_HIGH)) for epoch, array in maps.items()}
    last = own[EPOCHS[-1]]

    def panel(epoch, low, high, note):
        shown = np.clip((maps[epoch] - low) / max(float(high - low), 1e-6), 0, 1)
        rgb = colorize(orient_display(shown, grid.SHEET), "gray_r", grid.BAD, vmin=0.0, vmax=1.0)
        return (f"{RUN.name} epoch {epoch} | depth 0 | {note}: p1->0 ({low:.3f}), p99.5->1 ({high:.3f})", rgb, cv2.INTER_AREA, None)

    rows = [[panel(e, *own[e], "own scale") for e in EPOCHS], [panel(e, *last, f"epoch {EPOCHS[-1]} scale") for e in EPOCHS]]
    path = grid.OUT / "hercuenet_epochs_depth0.png"
    save_panel_grid(rows, mask.shape, str(path), gap_px=24)
    print(f"Saved {path}  (epochs {EPOCHS})", flush=True)


if __name__ == "__main__":
    main()
