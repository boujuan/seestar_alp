"""Sun-avoidance pre-flight safety primitives.

Pure helpers used by the goto pre-flight guard in `seestar_device.py`
and by the persistent toggle in the web UI. Dependency-free beyond
`ephem` (already a project dependency).

The Seestar's IMX585 sensor is destroyed within seconds by an unfiltered
slew to the sun, so any goto that targets the sun (or whose slew path
crosses it) is refused by default. The operator can disable the check
once the official magnetic ND solar filter is physically installed.

Conventions:
- Az is measured east of north, in degrees, in the range [0, 360).
- Altitude (el) is measured from the horizon, in degrees, in [-90, 90].
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import ephem


# Conservative defaults: 30 degree exclusion cone; sun must be at least
# 10 degrees below horizon for the check to short-circuit.
DEFAULT_MIN_SEPARATION_DEG = 30.0
DEFAULT_ALT_THRESHOLD_DEG = -10.0


# Pre-flight enable flag — process-global, runtime-toggleable from the
# web UI. Default ON; flipped OFF only when the operator explicitly
# acknowledges the solar filter is installed.
_enabled_lock = threading.Lock()
_enabled = True


def is_pre_flight_enabled() -> bool:
    with _enabled_lock:
        return _enabled


def set_pre_flight_enabled(value: bool) -> None:
    global _enabled
    with _enabled_lock:
        _enabled = bool(value)


@dataclass
class _Site:
    lat_deg: float
    lon_deg: float


def _site_from_config_or(lat_deg: Optional[float], lon_deg: Optional[float]) -> _Site:
    """Resolve site lat/lon, falling back to ``Config`` if not supplied."""
    if lat_deg is not None and lon_deg is not None:
        return _Site(float(lat_deg), float(lon_deg))
    from device.config import Config

    return _Site(float(Config.init_lat), float(Config.init_long))


def angular_separation(
    a_az_deg: float, a_el_deg: float, b_az_deg: float, b_el_deg: float
) -> float:
    """Great-circle angular separation between two (az, el) directions, in deg."""
    a_az = math.radians(a_az_deg)
    a_el = math.radians(a_el_deg)
    b_az = math.radians(b_az_deg)
    b_el = math.radians(b_el_deg)
    cos_sep = math.sin(a_el) * math.sin(b_el) + math.cos(a_el) * math.cos(
        b_el
    ) * math.cos(a_az - b_az)
    cos_sep = max(-1.0, min(1.0, cos_sep))
    return math.degrees(math.acos(cos_sep))


def compute_sun_altaz(
    *,
    lat_deg: Optional[float] = None,
    lon_deg: Optional[float] = None,
    when: Optional[datetime] = None,
) -> tuple[float, float]:
    """Sun (az, alt) in degrees as seen from the site at ``when`` (UTC now by default)."""
    site = _site_from_config_or(lat_deg, lon_deg)
    obs = ephem.Observer()
    obs.lat = str(site.lat_deg)
    obs.lon = str(site.lon_deg)
    if when is None:
        when = datetime.now(tz=timezone.utc)
    elif when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    obs.date = when.astimezone(timezone.utc).replace(tzinfo=None)
    sun = ephem.Sun()
    sun.compute(obs)
    return math.degrees(float(sun.az)), math.degrees(float(sun.alt))


def radec_to_topocentric_altaz(
    ra_hours: float,
    dec_deg: float,
    *,
    lat_deg: Optional[float] = None,
    lon_deg: Optional[float] = None,
    when: Optional[datetime] = None,
) -> tuple[float, float]:
    """Convert apparent (RA hours, Dec degrees) to topocentric (az, alt) deg."""
    site = _site_from_config_or(lat_deg, lon_deg)
    obs = ephem.Observer()
    obs.lat = str(site.lat_deg)
    obs.lon = str(site.lon_deg)
    if when is None:
        when = datetime.now(tz=timezone.utc)
    elif when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    obs.date = when.astimezone(timezone.utc).replace(tzinfo=None)
    body = ephem.FixedBody()
    body._ra = ephem.hours(ra_hours * math.pi / 12.0)
    body._dec = ephem.degrees(dec_deg * math.pi / 180.0)
    body._epoch = ephem.J2000
    body.compute(obs)
    az_deg = math.degrees(float(body.az)) % 360.0
    alt_deg = math.degrees(float(body.alt))
    return az_deg, alt_deg


def is_sun_safe(
    target_az_deg: float,
    target_el_deg: float,
    *,
    lat_deg: Optional[float] = None,
    lon_deg: Optional[float] = None,
    when: Optional[datetime] = None,
    min_separation_deg: float = DEFAULT_MIN_SEPARATION_DEG,
    alt_threshold_deg: float = DEFAULT_ALT_THRESHOLD_DEG,
) -> tuple[bool, str]:
    """Return ``(safe, reason)`` for pointing the optical axis at ``(az, el)``."""
    sun_az, sun_alt = compute_sun_altaz(
        lat_deg=lat_deg, lon_deg=lon_deg, when=when,
    )
    if sun_alt < alt_threshold_deg:
        return True, ""
    sep = angular_separation(target_az_deg, target_el_deg, sun_az, sun_alt)
    if sep < min_separation_deg:
        return False, (
            f"sun_avoidance: target separation {sep:.1f} deg < cone "
            f"{min_separation_deg:.1f} deg "
            f"(sun alt {sun_alt:.1f} deg, sun az {sun_az:.1f} deg)"
        )
    return True, ""


def check_path_crosses_sun(
    start_az_deg: float,
    start_el_deg: float,
    target_az_deg: float,
    target_el_deg: float,
    *,
    lat_deg: Optional[float] = None,
    lon_deg: Optional[float] = None,
    when: Optional[datetime] = None,
    min_separation_deg: float = DEFAULT_MIN_SEPARATION_DEG,
    alt_threshold_deg: float = DEFAULT_ALT_THRESHOLD_DEG,
    n_samples: int = 100,
) -> tuple[bool, float, str]:
    """Check if the alt-az slew path from start to target passes within the sun cone.

    Returns ``(crosses, min_sep_deg, reason)``. Az is interpolated along
    the shorter arc (handles 0 / 360 wrap). Always returns
    ``(False, 180.0, "")`` if the sun is below ``alt_threshold_deg``.
    """
    sun_az, sun_alt = compute_sun_altaz(
        lat_deg=lat_deg, lon_deg=lon_deg, when=when,
    )
    if sun_alt < alt_threshold_deg:
        return False, 180.0, ""

    daz = ((target_az_deg - start_az_deg + 540.0) % 360.0) - 180.0
    dalt = target_el_deg - start_el_deg

    min_sep = 180.0
    worst_az = start_az_deg
    worst_el = start_el_deg
    for i in range(n_samples + 1):
        t = i / n_samples
        az = (start_az_deg + t * daz) % 360.0
        el = start_el_deg + t * dalt
        sep = angular_separation(az, el, sun_az, sun_alt)
        if sep < min_sep:
            min_sep = sep
            worst_az = az
            worst_el = el

    if min_sep < min_separation_deg:
        return True, min_sep, (
            f"sun_avoidance: slew path passes {min_sep:.1f} deg from sun "
            f"(cone {min_separation_deg:.1f} deg; "
            f"sun az {sun_az:.1f} deg alt {sun_alt:.1f} deg; "
            f"closest point az {worst_az:.1f} deg el {worst_el:.1f} deg)"
        )
    return False, min_sep, ""


def evaluate_goto_safety(
    target_ra_hours: float,
    target_dec_deg: float,
    *,
    start_az_deg: Optional[float] = None,
    start_el_deg: Optional[float] = None,
    lat_deg: Optional[float] = None,
    lon_deg: Optional[float] = None,
    when: Optional[datetime] = None,
    min_separation_deg: float = DEFAULT_MIN_SEPARATION_DEG,
    alt_threshold_deg: float = DEFAULT_ALT_THRESHOLD_DEG,
) -> dict:
    """High-level pre-flight evaluation used by the goto guard.

    Returns a dict ``{safe, reason, target_alt_deg, target_az_deg,
    sun_alt_deg, sun_az_deg, min_path_sep_deg}``. ``safe`` is False if
    the target itself is within the cone OR if the slew path crosses it.
    """
    target_az, target_alt = radec_to_topocentric_altaz(
        target_ra_hours, target_dec_deg,
        lat_deg=lat_deg, lon_deg=lon_deg, when=when,
    )
    sun_az, sun_alt = compute_sun_altaz(
        lat_deg=lat_deg, lon_deg=lon_deg, when=when,
    )
    result = {
        "safe": True,
        "reason": "",
        "target_az_deg": target_az,
        "target_alt_deg": target_alt,
        "sun_az_deg": sun_az,
        "sun_alt_deg": sun_alt,
        "min_path_sep_deg": None,
    }
    if sun_alt < alt_threshold_deg:
        return result

    target_safe, target_reason = is_sun_safe(
        target_az, target_alt,
        lat_deg=lat_deg, lon_deg=lon_deg, when=when,
        min_separation_deg=min_separation_deg,
        alt_threshold_deg=alt_threshold_deg,
    )
    if not target_safe:
        result["safe"] = False
        result["reason"] = target_reason
        return result

    if start_az_deg is not None and start_el_deg is not None:
        crosses, min_sep, path_reason = check_path_crosses_sun(
            start_az_deg, start_el_deg, target_az, target_alt,
            lat_deg=lat_deg, lon_deg=lon_deg, when=when,
            min_separation_deg=min_separation_deg,
            alt_threshold_deg=alt_threshold_deg,
        )
        result["min_path_sep_deg"] = min_sep
        if crosses:
            result["safe"] = False
            result["reason"] = path_reason
    return result
