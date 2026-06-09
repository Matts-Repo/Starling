"""Batch recipe: YAML schema, validation, canonical hash."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

VALID_FITS = ("moments", "gauss1d", "gauss2p", "gauss2d")


@dataclass
class ScanEntry:
    file: str
    scan_id: str
    alias: str


@dataclass
class Recipe:
    output_dir: str
    scans: list = field(default_factory=list)  # list[ScanEntry]
    device: str = "auto"
    preprocess: dict = field(default_factory=dict)
    fits: list = field(default_factory=lambda: ["moments", "gauss1d"])
    fit_options: dict = field(default_factory=dict)
    timeseries: dict = field(default_factory=lambda: {"enabled": True})

    @classmethod
    def load(cls, path):
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw):
        errors = []
        if not isinstance(raw, dict):
            raise ValueError("recipe must be a YAML mapping")
        if "output_dir" not in raw:
            errors.append("missing required key: output_dir")
        scans = []
        for i, s in enumerate(raw.get("scans") or []):
            missing = {"file", "scan_id"} - set(s)
            if missing:
                errors.append(f"scans[{i}]: missing {sorted(missing)}")
                continue
            alias = s.get("alias") or f"scan_{i:03d}"
            scans.append(ScanEntry(file=s["file"], scan_id=str(s["scan_id"]), alias=alias))
        if not scans and not errors:
            errors.append("scans list is empty")
        aliases = [s.alias for s in scans]
        if len(set(aliases)) != len(aliases):
            errors.append("scan aliases must be unique")
        for fit in raw.get("fits", []):
            if fit not in VALID_FITS:
                errors.append(f"unknown fit '{fit}' (valid: {VALID_FITS})")
        if errors:
            raise ValueError("invalid recipe:\n  " + "\n  ".join(errors))

        return cls(
            output_dir=raw["output_dir"],
            scans=scans,
            device=raw.get("device", "auto"),
            preprocess=raw.get("preprocess") or {},
            fits=list(raw.get("fits", ["moments", "gauss1d"])),
            fit_options=raw.get("fit_options") or {},
            timeseries=raw.get("timeseries") or {"enabled": True},
        )

    def recipe_hash(self):
        """Hash of the processing definition (not the scan list or paths) —
        used to decide whether an existing output is reusable."""
        canon = json.dumps(
            {
                "preprocess": self.preprocess,
                "fits": sorted(self.fits),
                "fit_options": self.fit_options,
            },
            sort_keys=True,
        )
        return hashlib.sha256(canon.encode()).hexdigest()[:16]

    def output_path(self, alias):
        return str(Path(self.output_dir) / f"{alias}.h5")
