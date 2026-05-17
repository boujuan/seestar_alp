"""Web-driven satellite tracking session.

Wraps ``device.streaming_controller.track`` in a daemon thread so the
web UI can start / stop a track without blocking the request thread.
One global session at a time — second start while one is active returns
without doing anything.

Modelled on ``device.rotation_calibration.CalibrationSession`` (same
session-as-thread pattern) and follows the singleton accessor pattern
of ``device.sun_safety``.

The session does NOT touch the per-telescope ``Seestar`` object; it
talks to the mount through ``device.alpaca_client.AlpacaClient`` so it
works the same way ``scripts/trajectory/track.py`` does.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from astropy.coordinates import EarthLocation

from device.alpaca_client import AlpacaClient
from device.config import Config
from device.plant_limits import AzimuthLimits, CumulativeAzTracker
from device.reference_provider import JsonlECEFProvider
from device.streaming_controller import TickInfo, pre_check, track
from device.target_frame import MountFrame
from device.velocity_controller import PositionLogger, measure_altaz_timed


logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CALIBRATION_JSON = _REPO_ROOT / "device" / "mount_calibration.json"
_LOG_DIR = _REPO_ROOT / "auto_level_logs"


@dataclass
class SatelliteTrackStatus:
    """Public snapshot of the running session. Returned from /api/satellites/active."""

    active: bool = False
    pass_name: Optional[str] = None
    file: Optional[str] = None
    phase: str = "idle"           # idle | pre_check | waiting | tracking | finished
    dry_run: bool = False
    started_unix: Optional[float] = None
    finished_unix: Optional[float] = None
    pass_start_unix: Optional[float] = None
    pass_end_unix: Optional[float] = None
    # Last tick from streaming_controller (if any)
    last_tick: Optional[int] = None
    last_tick_unix: Optional[float] = None
    cur_alt_deg: Optional[float] = None
    cur_az_deg: Optional[float] = None
    ref_alt_deg: Optional[float] = None
    ref_az_deg: Optional[float] = None
    err_az_deg: Optional[float] = None
    err_el_deg: Optional[float] = None
    # Pre-check + outcome
    pre_check_feasible: Optional[bool] = None
    pre_check_notes: list[str] = field(default_factory=list)
    exit_reason: Optional[str] = None
    errors: list[str] = field(default_factory=list)


class SatelliteTrackSession:
    """One-shot session that drives streaming_controller.track from a thread."""

    def __init__(
        self,
        trajectory_path: Path,
        *,
        dry_run: bool = False,
        skip_precheck: bool = False,
        tick_dt: float = 0.5,
        latency_s: float = 0.4,
        tau_s: float = 0.348,
        kp_pos: float = 0.5,
        v_corr_max: float = 2.0,
        v_max: float = 6.0,
        el_max_deg: float = 85.0,
        max_duration_s: float = 1200.0,
        host: str = "127.0.0.1",
        port: int = 5555,
        device_id: int = 1,
    ) -> None:
        self.path = Path(trajectory_path)
        self.dry_run = dry_run
        self.skip_precheck = skip_precheck
        self.tick_dt = tick_dt
        self.latency_s = latency_s
        self.tau_s = tau_s
        self.kp_pos = kp_pos
        self.v_corr_max = v_corr_max
        self.v_max = v_max
        self.el_max_deg = el_max_deg
        self.max_duration_s = max_duration_s
        self.host = host
        self.port = port
        self.device_id = device_id

        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._status = SatelliteTrackStatus(
            file=str(self.path),
            dry_run=dry_run,
        )

    # ---------- lifecycle ----------

    def start(self) -> SatelliteTrackStatus:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self._snapshot_unlocked()
            self._status.active = True
            self._status.phase = "pre_check"
            self._status.started_unix = time.time()
            self._stop_evt.clear()
            self._thread = threading.Thread(
                target=self._run,
                daemon=True,
                name=f"sat-track:{self.path.stem}",
            )
        self._thread.start()
        logger.info("satellite-track session started: %s (dry_run=%s)",
                    self.path.name, self.dry_run)
        return self.status()

    def stop(self, timeout: float = 6.0) -> SatelliteTrackStatus:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        logger.info("satellite-track session stopped")
        return self.status()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def status(self) -> SatelliteTrackStatus:
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> SatelliteTrackStatus:
        # Return a shallow copy so external readers don't see torn state
        # mid-update (lock is held only briefly each tick).
        s = self._status
        return SatelliteTrackStatus(
            active=s.active,
            pass_name=s.pass_name,
            file=s.file,
            phase=s.phase,
            dry_run=s.dry_run,
            started_unix=s.started_unix,
            finished_unix=s.finished_unix,
            pass_start_unix=s.pass_start_unix,
            pass_end_unix=s.pass_end_unix,
            last_tick=s.last_tick,
            last_tick_unix=s.last_tick_unix,
            cur_alt_deg=s.cur_alt_deg,
            cur_az_deg=s.cur_az_deg,
            ref_alt_deg=s.ref_alt_deg,
            ref_az_deg=s.ref_az_deg,
            err_az_deg=s.err_az_deg,
            err_el_deg=s.err_el_deg,
            pre_check_feasible=s.pre_check_feasible,
            pre_check_notes=list(s.pre_check_notes),
            exit_reason=s.exit_reason,
            errors=list(s.errors),
        )

    def _on_tick(self, info: TickInfo) -> None:
        # Called from the track() loop's thread on every tick. Keep the
        # critical section short.
        with self._lock:
            self._status.last_tick = info.tick
            self._status.last_tick_unix = info.t_wall
            self._status.cur_alt_deg = info.cur_el_deg
            self._status.cur_az_deg = info.cur_cum_az_deg
            self._status.ref_alt_deg = info.eff_ref_el_deg
            self._status.ref_az_deg = info.eff_ref_az_cum_deg
            self._status.err_az_deg = info.err_az_deg
            self._status.err_el_deg = info.err_el_deg

    # ---------- thread body ----------

    def _run(self) -> None:
        try:
            self._do_run()
        except Exception as exc:
            logger.exception("satellite-track session crashed")
            with self._lock:
                self._status.errors.append(f"crash: {exc}")
                self._status.phase = "finished"
                self._status.exit_reason = "crash"
                self._status.active = False
                self._status.finished_unix = time.time()

    def _do_run(self) -> None:
        # Build observer site + mount frame. Use scripts.trajectory.observer's
        # build_site (returns a full ObserverSite with ecef_xyz, ENU rotation,
        # etc.) — anchored to the Seestar's configured lat/long.
        from scripts.trajectory.observer import build_site
        lat = float(Config.init_lat)
        lon = float(Config.init_long)
        alt = float(getattr(Config, "init_height", 0.0) or 0.0)
        loc = EarthLocation.from_geodetic(lon=lon, lat=lat, height=alt)
        site = build_site(lat_deg=lat, lon_deg=lon, alt_m=alt)
        if _CALIBRATION_JSON.exists():
            try:
                mount_frame = MountFrame.from_calibration_json(_CALIBRATION_JSON, site)
            except Exception:
                mount_frame = MountFrame.from_identity_enu(site)
        else:
            mount_frame = MountFrame.from_identity_enu(site)

        # Load provider
        provider = JsonlECEFProvider(self.path, mount_frame)
        header = provider.header
        pass_name = (
            header.get("name") or header.get("callsign")
            or header.get("id") or self.path.stem
        )
        t_start_traj, t_end_traj = provider.valid_range()
        with self._lock:
            self._status.pass_name = pass_name
            self._status.pass_start_unix = float(t_start_traj)
            self._status.pass_end_unix = float(t_end_traj)

        # Pre-check 0: refuse if mount is in EQ mode. The streaming
        # controller commands az/el directional moves; in EQ mode the
        # physical axes are RA/Dec and the same command would slew the
        # mount in wildly wrong directions. The Seestar firmware exposes
        # mount.equ_mode in get_device_state.
        try:
            cli_probe = AlpacaClient(self.host, self.port, self.device_id)
            dev_state = cli_probe.method_sync("get_device_state").get("result", {})
            if dev_state.get("mount", {}).get("equ_mode") is True:
                with self._lock:
                    self._status.phase = "finished"
                    self._status.exit_reason = "eq_mode_unsupported"
                    self._status.errors.append(
                        "Mount is in EQ mode — satellite tracking only "
                        "supports alt-az for now. Switch the Seestar to "
                        "alt-az mode and try again."
                    )
                    self._status.active = False
                    self._status.finished_unix = time.time()
                return
        except Exception:
            # If we can't query mount state, log it but continue —
            # the user may be on older firmware that doesn't expose it.
            logger.debug("equ_mode probe failed", exc_info=True)

        # Pre-check 1: cable wrap, el-limit, FF saturation
        az_limits = AzimuthLimits.load()
        pre = pre_check(
            provider, az_limits=az_limits,
            el_max_deg=self.el_max_deg, el_min_deg=-self.el_max_deg,
            tick_dt=self.tick_dt, tau_s=self.tau_s, v_max=self.v_max,
        )
        with self._lock:
            self._status.pre_check_feasible = bool(pre.feasible)
            self._status.pre_check_notes = list(pre.notes)
        if (not pre.feasible) and (not self.skip_precheck):
            with self._lock:
                self._status.phase = "finished"
                self._status.exit_reason = "pre_check_failed"
                self._status.errors.append(
                    "pre-check refused — set skip_precheck=true to override"
                )
                self._status.active = False
                self._status.finished_unix = time.time()
            return

        # Connect mount
        cli = AlpacaClient(self.host, self.port, self.device_id)
        try:
            alt0, az0_wrapped, _ = measure_altaz_timed(cli, loc)
        except Exception as exc:
            with self._lock:
                self._status.phase = "finished"
                self._status.exit_reason = "mount_error"
                self._status.errors.append(f"mount connect failed: {exc}")
                self._status.active = False
                self._status.finished_unix = time.time()
            return

        # Position logger (writes JSONL for post-hoc analysis)
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = _LOG_DIR / (
            f"{time.strftime('%Y-%m-%dT%H-%M-%S')}.satellite-{self.path.stem}.jsonl"
        )
        position_logger = PositionLogger(
            cli, loc, log_path, poll_interval_s=self.tick_dt,
        )
        position_logger.start()
        try:
            position_logger.set_phase("track_init")
            position_logger.mark_event(
                "track_start",
                trajectory=str(self.path),
                pass_name=pass_name,
                dry_run=self.dry_run,
            )

            tracker = CumulativeAzTracker.load_or_fresh(
                current_wrapped_az_deg=az0_wrapped,
            )

            # Wait until t_start; honor stop_signal while waiting.
            with self._lock:
                self._status.phase = "waiting"
            while True:
                if self._stop_evt.is_set():
                    with self._lock:
                        self._status.phase = "finished"
                        self._status.exit_reason = "stop_signal"
                        self._status.active = False
                        self._status.finished_unix = time.time()
                    return
                now = time.time()
                if now + self.latency_s >= t_start_traj:
                    break
                time.sleep(min(self.tick_dt, max(0.05, t_start_traj - now - self.latency_s)))

            with self._lock:
                self._status.phase = "tracking"

            result = track(
                cli, provider,
                tick_dt=self.tick_dt, latency_s=self.latency_s,
                tau_s=self.tau_s, kp_pos=self.kp_pos,
                v_corr_max=self.v_corr_max, v_max=self.v_max,
                az_limits=az_limits, az_tracker=tracker,
                position_logger=position_logger,
                stop_signal=self._stop_evt,
                max_duration_s=self.max_duration_s,
                el_max_deg=self.el_max_deg, el_min_deg=-self.el_max_deg,
                dry_run=self.dry_run,
                tick_callback=self._on_tick,
            )
            with self._lock:
                self._status.phase = "finished"
                self._status.exit_reason = result.exit_reason
                self._status.errors.extend(result.errors)
                self._status.active = False
                self._status.finished_unix = time.time()
        finally:
            try:
                position_logger.mark_event("track_end")
                position_logger.stop()
            except Exception:
                pass


# Minimal shim so MountFrame.from_* works without importing the heavier
# observer module here (and to avoid circular-import surprises).
@dataclass
class _SiteShim:
    lat_deg: float
    lon_deg: float
    alt_m: float


# ---------- module-level singleton ----------

_session: Optional[SatelliteTrackSession] = None
_session_lock = threading.Lock()


def get_satellite_session() -> Optional[SatelliteTrackSession]:
    with _session_lock:
        return _session


def set_satellite_session(s: Optional[SatelliteTrackSession]) -> None:
    global _session
    with _session_lock:
        _session = s


def start_session(path: Path, **kwargs) -> tuple[SatelliteTrackSession, bool]:
    """Start a new session if none is active. Returns (session, started_new)."""
    with _session_lock:
        global _session
        if _session is not None and _session.is_alive():
            return _session, False
        _session = SatelliteTrackSession(path, **kwargs)
    _session.start()
    return _session, True


def stop_session(timeout: float = 6.0) -> Optional[SatelliteTrackStatus]:
    s = get_satellite_session()
    if s is None:
        return None
    return s.stop(timeout=timeout)


__all__ = [
    "SatelliteTrackSession",
    "SatelliteTrackStatus",
    "get_satellite_session",
    "set_satellite_session",
    "start_session",
    "stop_session",
]
