"""Small nonlinear lookup fallback; replace with role 2 through the same API."""

import json
from pathlib import Path

from .core import Drive


class TableDrive:
    def __init__(self, path):
        self.data = json.loads(Path(path).read_text(encoding="utf-8"))

    def predict(self, command: int, speed_mps: float) -> Drive:
        bins = self.data["speed_bins_mps"]
        row = self.data["acceleration_mps2"].get(str(command), self.data["acceleration_mps2"]["0"])
        i = max(0, min(len(bins)-2, next((i-1 for i, v in enumerate(bins) if v > speed_mps), len(bins)-2)))
        frac = min(1.0, max(0.0, (speed_mps-bins[i])/(bins[i+1]-bins[i])))
        a = row[i]*(1-frac)+row[i+1]*frac
        return Drive(a, self.data["sigma_mps2"].get(str(command), 0.5))
