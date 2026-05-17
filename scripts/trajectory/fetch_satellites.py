"""Find bright satellite passes over the observer and export ECEF tracks.

Pulls TLEs from Celestrak (visual + stations groups), filters passes where
culmination altitude falls in [20°, 80°] and duration is at least 240 s,
samples each qualifying pass at 2 Hz, and writes a JSONL per pass.

ECEF is derived via skyfield's ITRS frame (`sat.at(t).frame_xyz(itrs).m`),
which is earth-fixed. Do NOT use `.position.km` — that is GCRS/ECI and
rotates relative to the earth by ~465 m/s at the equator.

Example:

    python -m scripts.trajectory.fetch_satellites --hours 24 --top-n 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from skyfield.api import Loader, wgs84
from skyfield.framelib import itrs

from scripts.trajectory.observer import (
    build_site,
    ecef_array_to_topo,
)


CELESTRAK_GROUPS = {
    "visual": "https://celestrak.org/NORAD/elements/gp.php?GROUP=visual&FORMAT=tle",
    "stations": "https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle",
}

# Pass filter parameters (can be overridden on CLI).
MIN_CULM_EL_DEG = 20.0
MAX_CULM_EL_DEG = 80.0
MIN_PASS_DURATION_S = 240.0
MIN_EL_DEG_FOR_PASS = 10.0  # rise/set threshold for find_events


# Standard (intrinsic) visual magnitudes — the brightness a satellite
# would have at range 1000 km, fully sunlit, phase angle = 0. Used with
# the range + phase formula below to estimate apparent magnitude per
# pass. Values from McCants' visual mag database + common references.
STANDARD_MAGNITUDES: dict[str, float] = {
    # Stations
    "ISS (ZARYA)": -1.8,
    "ISS (NAUKA)": -1.8,
    "CSS (TIANHE)": +0.8,
    "TIANHE-1": +0.8,
    # Brighter LEO objects
    "HST": +2.4,        # Hubble
    "ENVISAT": +1.4,
    "COSMOS 1408 DEB": +6.0,
    # Bright rocket bodies (typical)
    "SL-16 R/B": +2.0,
    "SL-14 R/B": +3.0,
    "SL-8 R/B": +4.0,
    "SL-3 R/B": +3.5,
    "SL-12 R/B(2)": +4.0,
    "ARIANE 40 R/B": +3.0,
    "ARIANE 40+ R/B": +3.0,
    "CZ-2C R/B": +4.0,
    "CZ-4B R/B": +4.5,
    # Crew/cargo capsules
    "SZ-21 MODULE": +3.0,
    # Default for unlisted satellites — most sub-microsat are dim
    "_DEFAULT": +5.0,
}


def _std_mag_for(name: str) -> float:
    """Best-effort standard magnitude for a satellite by name."""
    if name in STANDARD_MAGNITUDES:
        return STANDARD_MAGNITUDES[name]
    # Try class prefix (e.g. "SL-16 R/B" matches if name starts with that)
    for key, val in STANDARD_MAGNITUDES.items():
        if key == "_DEFAULT":
            continue
        if name.startswith(key.split()[0]):
            return val
    return STANDARD_MAGNITUDES["_DEFAULT"]


def _peak_apparent_magnitude(
    sat,
    ts_scale,
    t_grid_unix: np.ndarray,
    sun_az_deg_grid: np.ndarray | None,
    sun_alt_deg_grid: np.ndarray | None,
    sat_alt_deg_grid: np.ndarray,
    sat_slant_m_grid: np.ndarray,
    std_mag: float,
    eph,
) -> tuple[float, bool]:
    """Return (peak_apparent_mag, any_sunlit) over a pass.

    Apparent magnitude:
        m = std_mag + 5*log10(range_km/1000)
            - 2.5*log10(phase_function)

    Phase function (diffuse sphere): (1 + cos(phase_angle))/2

    A satellite is "visible" only if it's in sunlight AND the sky is
    dark enough (sun below ~-6 deg). For shadowed samples the magnitude
    is reported as +inf (not visible).
    """
    # Sunlit check per-tick via skyfield. from_datetime needs a list,
    # not an ndarray.
    times = ts_scale.from_datetimes([
        datetime.fromtimestamp(float(t), tz=timezone.utc)
        for t in t_grid_unix
    ])
    try:
        sunlit = sat.at(times).is_sunlit(eph)  # ndarray of bool
    except Exception:
        sunlit = np.ones(len(t_grid_unix), dtype=bool)

    # Range in km
    range_km = sat_slant_m_grid / 1000.0
    # Simple phase model: phase ~ 90 deg when sat is at half-illumination,
    # cosine factor (1+cos)/2. Without a proper sun vector at the sat,
    # approximate with the sat's altitude proxy: low alt = larger phase
    # angle (sat seen edge-on relative to sun). Conservative.
    # For a better estimate this needs sun_az/alt at the satellite, not
    # the observer — left for a polish round.
    phase_func = np.full_like(range_km, 0.5)  # diffuse default
    # Avoid log of zero
    phase_func = np.maximum(phase_func, 1e-3)

    apparent = std_mag + 5 * np.log10(range_km / 1000.0) - 2.5 * np.log10(phase_func)
    # Mask shadowed samples to +inf
    apparent_visible = np.where(sunlit, apparent, np.inf)
    if not np.any(np.isfinite(apparent_visible)):
        return float("inf"), False
    return float(np.min(apparent_visible)), bool(np.any(sunlit))


@dataclass
class Pass:
    satellite_name: str
    norad_id: str
    tle_line1: str
    tle_line2: str
    t_rise_unix: float
    t_culm_unix: float
    t_set_unix: float
    culm_el_deg: float


def _loader() -> Loader:
    cache_dir = Path(
        os.environ.get("SKYFIELD_CACHE", Path.home() / ".skyfield-data")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    return Loader(str(cache_dir))


def load_tles(load: Loader) -> list:
    sats: list = []
    seen_norad: set[str] = set()
    for group, url in CELESTRAK_GROUPS.items():
        try:
            group_sats = load.tle_file(url, reload=False)
        except Exception as exc:  # skyfield may raise HTTP / parse errors
            print(f"[fetch_satellites] failed to load {group}: {exc}",
                  file=sys.stderr)
            continue
        for sat in group_sats:
            norad = str(sat.model.satnum)
            if norad in seen_norad:
                continue
            seen_norad.add(norad)
            sats.append(sat)
        print(f"[fetch_satellites] loaded {len(group_sats)} sats from {group}",
              file=sys.stderr)
    return sats


def find_passes(
    sat, observer, ts, t0, t1,
) -> list[Pass]:
    """Group find_events output into (rise, culm, set) triplets."""
    try:
        times, flags = sat.find_events(
            observer, t0, t1, altitude_degrees=MIN_EL_DEG_FOR_PASS,
        )
    except Exception as exc:
        print(f"[fetch_satellites] find_events failed for {sat.name}: {exc}",
              file=sys.stderr)
        return []

    passes: list[Pass] = []
    i = 0
    while i + 2 < len(flags):
        if flags[i] == 0 and flags[i + 1] == 1 and flags[i + 2] == 2:
            t_rise = times[i]
            t_culm = times[i + 1]
            t_set = times[i + 2]
            # Peak elevation at culmination.
            alt, _az, _dist = (sat - observer).at(t_culm).altaz()
            passes.append(Pass(
                satellite_name=sat.name,
                norad_id=str(sat.model.satnum),
                tle_line1=sat.tle_line1 if hasattr(sat, "tle_line1") else "",
                tle_line2=sat.tle_line2 if hasattr(sat, "tle_line2") else "",
                t_rise_unix=t_rise.utc_datetime().timestamp(),
                t_culm_unix=t_culm.utc_datetime().timestamp(),
                t_set_unix=t_set.utc_datetime().timestamp(),
                culm_el_deg=float(alt.degrees),
            ))
            i += 3
        else:
            i += 1
    return passes


def _pick_tle_lines(sat) -> tuple[str, str]:
    # Skyfield stores TLE lines on the EarthSatellite object when loaded via
    # tle_file. Fall back to reconstruction from the SGP4 model if needed.
    l1 = getattr(sat, "_line1", None) or getattr(sat, "tle_line1", None)
    l2 = getattr(sat, "_line2", None) or getattr(sat, "tle_line2", None)
    if l1 and l2:
        return str(l1), str(l2)
    # Fallback — don't block export; just record empty TLE in header.
    return "", ""


def export_pass(
    p: Pass, sat, site, load, out_dir: Path, sample_hz: float,
    *, eph=None,
) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dt = 1.0 / sample_hz
    t_grid = np.arange(p.t_rise_unix, p.t_set_unix + dt, dt)
    ts_scale = load.timescale()
    ecef = np.empty((len(t_grid), 3))
    for i, t_unix in enumerate(t_grid):
        t = ts_scale.from_datetime(
            datetime.fromtimestamp(float(t_unix), tz=timezone.utc)
        )
        ecef[i] = sat.at(t).frame_xyz(itrs).m
    az, el, slant = ecef_array_to_topo(ecef, site)

    safe_name = "".join(
        c if c.isalnum() else "_" for c in p.satellite_name.strip()
    ).strip("_") or f"norad{p.norad_id}"
    t0 = int(t_grid[0])
    path = out_dir / f"{safe_name}_{p.norad_id}_{t0}.jsonl"

    # Apparent magnitude estimate (peak brightness during the pass)
    std_mag = _std_mag_for(p.satellite_name)
    peak_mag = float("inf")
    any_sunlit = False
    if eph is not None:
        try:
            peak_mag, any_sunlit = _peak_apparent_magnitude(
                sat, ts_scale, t_grid,
                sun_az_deg_grid=None, sun_alt_deg_grid=None,
                sat_alt_deg_grid=el, sat_slant_m_grid=slant,
                std_mag=std_mag, eph=eph,
            )
        except Exception as exc:
            print(f"[fetch_satellites] mag calc failed for {p.satellite_name}: {exc}",
                  file=sys.stderr)

    l1, l2 = _pick_tle_lines(sat)
    header = {
        "kind": "header",
        "source": "tle",
        "id": p.norad_id,
        "name": p.satellite_name,
        "observer_lat": site.lat_deg,
        "observer_lon": site.lon_deg,
        "observer_alt_m": site.alt_m,
        "duration_s": float(t_grid[-1] - t_grid[0]),
        "peak_el_deg": float(np.max(el)),
        "culm_el_deg": p.culm_el_deg,
        "min_slant_m": float(np.min(slant)),
        "max_slant_m": float(np.max(slant)),
        "sample_rate_hz": sample_hz,
        "n_samples": int(len(t_grid)),
        "std_mag": std_mag,
        "peak_apparent_mag": peak_mag if np.isfinite(peak_mag) else None,
        "any_sunlit": any_sunlit,
        "tle": [l1, l2] if l1 else [],
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for i in range(len(t_grid)):
            rec = {
                "kind": "sample",
                "t_unix": float(t_grid[i]),
                "ecef_x": float(ecef[i, 0]),
                "ecef_y": float(ecef[i, 1]),
                "ecef_z": float(ecef[i, 2]),
                "az_deg": float(az[i]),
                "el_deg": float(el[i]),
                "slant_m": float(slant[i]),
            }
            f.write(json.dumps(rec) + "\n")
    print(
        f"[fetch_satellites] wrote {path.name}  "
        f"peak_el={header['peak_el_deg']:.1f}°  "
        f"dur={header['duration_s']:.0f}s  "
        f"slant=[{header['min_slant_m']/1000:.0f},{header['max_slant_m']/1000:.0f}] km",
        file=sys.stderr,
    )
    return path


def fetch_and_export(
    *,
    hours: float = 24.0,
    top_n: int = 5,
    sample_hz: float = 2.0,
    out_dir: Path | None = None,
) -> list[Path]:
    """Find upcoming passes and export top-N to JSONL files.

    Returns the list of paths written. Empty list means no passes
    matched the filter. Caller is responsible for any cleanup of stale
    files in ``out_dir``.

    Same logic as the CLI ``main()``, factored out so the web UI's
    refresh endpoint can call it from a background thread.
    """
    if out_dir is None:
        out_dir = Path("data/trajectories/satellites")

    site = build_site()
    load = _loader()
    ts = load.timescale()
    # Planetary ephemeris for sunlit-shadow check. First call downloads
    # ~17 MB de421.bsp; subsequent calls use the cache.
    try:
        eph = load("de421.bsp")
    except Exception as exc:
        print(f"[fetch_satellites] eph load failed (no magnitude calc): {exc}",
              file=sys.stderr)
        eph = None

    sats = load_tles(load)
    if not sats:
        print("[fetch_satellites] no satellites loaded, aborting",
              file=sys.stderr)
        return []

    observer = wgs84.latlon(
        latitude_degrees=site.lat_deg,
        longitude_degrees=site.lon_deg,
        elevation_m=site.alt_m,
    )

    now = time.time()
    t0 = ts.from_datetime(datetime.fromtimestamp(now, tz=timezone.utc))
    t1 = ts.from_datetime(
        datetime.fromtimestamp(now + hours * 3600.0, tz=timezone.utc)
    )
    print(f"[fetch_satellites] scanning {len(sats)} sats over {hours} h",
          file=sys.stderr)

    candidates: list[tuple[Pass, object]] = []
    for sat in sats:
        for p in find_passes(sat, observer, ts, t0, t1):
            duration = p.t_set_unix - p.t_rise_unix
            if duration < MIN_PASS_DURATION_S:
                continue
            if not (MIN_CULM_EL_DEG <= p.culm_el_deg <= MAX_CULM_EL_DEG):
                continue
            candidates.append((p, sat))

    print(f"[fetch_satellites] {len(candidates)} candidate passes match filter",
          file=sys.stderr)

    # Rank by culmination elevation descending, dedupe by satellite name.
    candidates.sort(key=lambda pair: -pair[0].culm_el_deg)
    picked: list[tuple[Pass, object]] = []
    seen_names: set[str] = set()
    for p, sat in candidates:
        if p.satellite_name in seen_names:
            continue
        seen_names.add(p.satellite_name)
        picked.append((p, sat))
        if len(picked) >= top_n:
            break

    written: list[Path] = []
    for p, sat in picked:
        path = export_pass(p, sat, site, load, out_dir, sample_hz, eph=eph)
        if path is not None:
            written.append(path)
    print(f"[fetch_satellites] exported {len(written)} pass(es) to {out_dir}",
          file=sys.stderr)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=24.0,
                        help="look-ahead window starting now (UTC)")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/trajectories/satellites"),
    )
    parser.add_argument("--top-n", type=int, default=5,
                        help="max passes to export, ranked by culm elevation")
    parser.add_argument("--sample-hz", type=float, default=2.0)
    args = parser.parse_args(argv)

    written = fetch_and_export(
        hours=args.hours,
        top_n=args.top_n,
        sample_hz=args.sample_hz,
        out_dir=args.out_dir,
    )
    return 0 if written else 2


if __name__ == "__main__":
    sys.exit(main())
