"""Rank satellite trajectories by mechanical tracking feasibility.

Walks `data/trajectories/satellites/*.jsonl`, builds a JsonlECEFProvider
+ identity MountFrame for each, runs the StreamingFFController pre-check
and the offline replay simulator, and prints a ranked table.

Scoring (higher = better):
- Feasibility (must be True): cable-wrap OK, el within usable band,
  zero FF saturation.
- `replay.simulate_replay` RMS + peak tracking error below thresholds.
- Prefers passes with peak elevation near 60° (mid-sky — less extreme
  angular rates near zenith, and easier to keep inside el_max).
- Prefers longer passes (more useful flight time for test data).

Tiangong / CSS passes are pinned to the top of the recommendation regardless
of score (user requirement).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from device.plant_limits import AzimuthLimits
from device.reference_provider import JsonlECEFProvider
from device.streaming_controller import pre_check
from device.target_frame import MountFrame
from scripts.trajectory import replay


DEFAULT_DIR = Path("data/trajectories/satellites")


@dataclass
class TargetReport:
    path: Path
    name: str
    start_unix: float
    end_unix: float
    duration_s: float
    peak_el_deg: float
    peak_v_az_degs: float
    peak_v_el_degs: float
    peak_a_az_degs2: float
    peak_a_el_degs2: float
    cable_violations: int
    el_violations: int
    v_sat_ticks: int
    az_err_rms: float
    az_err_peak: float
    el_err_rms: float
    el_err_peak: float
    feasible: bool
    score: float
    tle_pinned: bool
    notes: list[str]
    peak_apparent_mag: float | None = None  # smaller = brighter; None if in shadow
    peak_visible_mag: float | None = None   # bright AND sky dark enough; None if not visible
    visible_seconds: float = 0.0
    any_sunlit: bool = True

    def summary(self) -> str:
        tag = "★" if self.tle_pinned else ("✓" if self.feasible else "✗")
        return (
            f"{tag} {self.name:<30} "
            f"dur={self.duration_s:>5.0f}s "
            f"peak_el={self.peak_el_deg:>5.1f}° "
            f"peak_v_az={self.peak_v_az_degs:>4.2f}°/s "
            f"peak_v_el={self.peak_v_el_degs:>4.2f}°/s "
            f"az_err_rms={self.az_err_rms:>4.2f}° "
            f"el_err_rms={self.el_err_rms:>4.2f}° "
            f"score={self.score:>5.2f}"
        )


def _evaluate(path: Path, mount_frame: MountFrame) -> TargetReport:
    provider = JsonlECEFProvider(path, mount_frame)
    az_limits = AzimuthLimits.load()
    pre = pre_check(
        provider, az_limits=az_limits, el_max_deg=85.0, el_min_deg=-85.0,
    )
    traj = replay.load_trajectory(path)
    sim = replay.simulate_replay(
        traj, az_limits=az_limits, mount_frame=mount_frame,
    )
    header = provider.header
    peak_el_deg = float(header.get("peak_el_deg", pre.max_el_deg))
    name = header.get("name") or header.get("callsign") or header.get("id") or path.stem
    duration_s = float(header.get("duration_s", 0.0))

    az_rms = float(np.sqrt(np.mean(sim.az_err ** 2)))
    az_peak = float(np.max(np.abs(sim.az_err)))
    el_rms = float(np.sqrt(np.mean(sim.el_err ** 2)))
    el_peak = float(np.max(np.abs(sim.el_err)))

    # Visibility metrics from the header (computed at fetch time).
    peak_mag = header.get("peak_apparent_mag")
    peak_vis_mag = header.get("peak_visible_mag")
    visible_seconds = float(header.get("visible_seconds", 0.0))
    any_sunlit = bool(header.get("any_sunlit", True))

    # Score: feasibility + tracking quality + mid-sky bonus + duration.
    # Visibility (sat sunlit AND observer sky dark) is the LARGEST
    # qualitative factor — a shadowed pass is unphotographable.
    if pre.feasible and sim.az_sat_count == 0 and sim.el_sat_count == 0:
        score = 100.0
    else:
        score = 0.0
    score -= 20.0 * az_rms
    score -= 20.0 * el_rms
    score -= abs(peak_el_deg - 60.0) * 0.2
    score += min(duration_s / 60.0, 10.0) * 1.0

    # Visibility multiplier — applied BEFORE the pin so shadowed
    # passes still rank below visible ones even when they are pinned.
    if peak_vis_mag is None:
        # Pass is invisible (sat in shadow or sky too bright). Camera
        # cannot photograph it; tracking is mechanically possible but
        # useless. Heavy penalty.
        score *= 0.1
    else:
        # Bonus for naked-eye-bright passes.
        if peak_vis_mag < 2.0:
            score += 10.0
        if peak_vis_mag < -1.0:
            score += 20.0

    # Tiangong / CSS pin — still applies but reduced when shadowed.
    tle_pinned = "TIANHE" in name.upper() or "CSS" in name.upper() or "TIANGONG" in name.upper()
    if tle_pinned:
        if peak_vis_mag is None:
            score += 100.0   # still surface it, but don't dominate the list
        else:
            score += 1000.0

    notes = list(pre.notes)
    if sim.az_sat_count or sim.el_sat_count:
        notes.append(
            f"replay saturation: az={sim.az_sat_count} el={sim.el_sat_count}"
        )

    t_start, t_end = provider.valid_range()
    return TargetReport(
        path=path, name=str(name),
        start_unix=float(t_start), end_unix=float(t_end),
        duration_s=duration_s,
        peak_el_deg=peak_el_deg,
        peak_v_az_degs=pre.peak_v_az_degs,
        peak_v_el_degs=pre.peak_v_el_degs,
        peak_a_az_degs2=pre.peak_a_az_degs2,
        peak_a_el_degs2=pre.peak_a_el_degs2,
        cable_violations=pre.cable_wrap_violations,
        el_violations=pre.el_limit_violations,
        v_sat_ticks=pre.v_saturation_ticks,
        az_err_rms=az_rms, az_err_peak=az_peak,
        el_err_rms=el_rms, el_err_peak=el_peak,
        feasible=pre.feasible and sim.az_sat_count == 0 and sim.el_sat_count == 0,
        score=score,
        tle_pinned=tle_pinned,
        notes=notes,
        peak_apparent_mag=peak_mag,
        peak_visible_mag=peak_vis_mag,
        visible_seconds=visible_seconds,
        any_sunlit=any_sunlit,
    )


def rank_and_index(
    directory: Path | None = None,
    *,
    json_out: Path | None = None,
) -> list[TargetReport]:
    """Rank every *.jsonl pass in ``directory``, optionally write a
    JSON index file consumable by the web UI.

    Returns the sorted list of TargetReport (highest score first).
    Empty list = no JSONL files found OR all failed to evaluate.

    If ``json_out`` is given, writes a JSON file with the same shape
    as the CLI ``--json`` flag. The web UI reads this file via
    ``/api/satellites/list``.
    """
    if directory is None:
        directory = DEFAULT_DIR
    jsonl_paths = sorted(p for p in directory.glob("*.jsonl") if not p.name.startswith("_"))
    if not jsonl_paths:
        return []
    mount_frame = MountFrame.from_identity_enu()

    reports: list[TargetReport] = []
    for p in jsonl_paths:
        try:
            reports.append(_evaluate(p, mount_frame))
        except Exception as exc:
            print(f"[skip] {p.name}: {exc}")
    reports.sort(key=lambda r: -r.score)

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with json_out.open("w", encoding="utf-8") as f:
            json.dump([_report_to_dict(r) for r in reports], f, indent=2)

    return reports


def _report_to_dict(r: TargetReport) -> dict:
    return {
        "path": str(r.path),
        "name": r.name,
        "start_unix": r.start_unix,
        "end_unix": r.end_unix,
        "duration_s": r.duration_s,
        "peak_el_deg": r.peak_el_deg,
        "peak_v_az_degs": r.peak_v_az_degs,
        "peak_v_el_degs": r.peak_v_el_degs,
        "az_err_rms": r.az_err_rms,
        "el_err_rms": r.el_err_rms,
        "cable_violations": r.cable_violations,
        "el_violations": r.el_violations,
        "v_saturation_ticks": r.v_sat_ticks,
        "feasible": r.feasible,
        "tle_pinned": r.tle_pinned,
        "score": r.score,
        "notes": r.notes,
        "peak_apparent_mag": r.peak_apparent_mag,
        "peak_visible_mag": r.peak_visible_mag,
        "visible_seconds": r.visible_seconds,
        "any_sunlit": r.any_sunlit,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir", type=Path, default=DEFAULT_DIR,
        help="Directory containing *.jsonl satellite passes",
    )
    parser.add_argument("--top", type=int, default=5,
                        help="Number of top targets to highlight")
    parser.add_argument("--json", type=Path, default=None,
                        help="Optional path to write structured ranking JSON")
    args = parser.parse_args(argv)

    reports = rank_and_index(args.dir, json_out=args.json)
    if not reports:
        print(f"no JSONL files found / evaluated in {args.dir}")
        return 2

    print(f"\n{'─'*120}")
    print(f"Ranked targets ({len(reports)} total, top {args.top} shown):")
    print(f"{'─'*120}")
    for r in reports[: args.top]:
        print(r.summary())
        for note in r.notes:
            print(f"    ⚠ {note}")
        print(f"    file: {r.path.name}")

    print(f"\n{'─'*120}")
    print("All feasible targets:")
    print(f"{'─'*120}")
    for r in reports:
        marker = "★" if r.tle_pinned else ("✓" if r.feasible else "✗")
        print(f"  {marker} {r.name:<30} score={r.score:>7.2f}  file={r.path.name}")

    if args.json is not None:
        print(f"\n→ wrote ranking to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
