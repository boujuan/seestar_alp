"""Firmware error-flag monitor.

The Seestar firmware emits structured events when it rejects or fails an
operation — most usefully::

    {"Event": "ScopeTrack", "state": "off", "tracking": false,
     "error": "below horizon", "code": 270, ...}

These events are silently swallowed by the seestar_alp event ingestion
loop; the symptom an operator sees is a goto that "doesn't move" with
no explanation. The result is a long debugging session that ends with
"oh, the target was below the horizon."

This module:

1. Defines :class:`FirmwareError` — a normalized snapshot of a firmware
   error event.
2. Defines :class:`FirmwareErrorMonitor` — a thread-safe holder of the
   most recent user-actionable error, with an inferred-error path for
   silent-fail symptoms (goto submitted but AutoGoto never fired).
3. Exposes module-level :func:`get_firmware_error_monitor` /
   :func:`set_firmware_error_monitor` accessors mirroring the existing
   :func:`device.sun_safety.get_sun_monitor` pattern.

The seestar_alp event ingestion calls :meth:`FirmwareErrorMonitor.observe`
with every firmware ``Event:`` payload. Most events are ignored; only the
user-actionable codes are surfaced.

The front-end calls :meth:`get_pending` to get the most recent unread
error and :meth:`dismiss` to clear it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional


# Whitelist of firmware error codes that warrant a UI banner. Codes
# that are too common to be useful (e.g. plate-solve failures while
# indoors) are deliberately excluded.
#
# When the firmware emits one of these in an event with state in
# {"fail", "off"}, the monitor records the error.
USER_ACTIONABLE_CODES: dict[int, str] = {
    270: "Target is below the horizon — pick an above-horizon target.",
    207: "Mount couldn't complete the operation. Try Stop / Unstick Goto.",
    253: "Operation aborted.",
    527: (
        "Target is too close to the zenith for an alt-az mount "
        "— pick a target below ~85° altitude."
    ),
    501: "Mount goto failed. See AutoGoto box for details.",
}


# Events whose error flags we want to inspect. Limiting the scan to
# this set avoids surfacing trivia from event types we don't model.
_WATCHED_EVENT_NAMES = frozenset({
    "ScopeTrack",
    "ScopeGoto",
    "AutoGoto",
    "AutoGotoStep",
})


# Inferred-error code: not a real firmware code, used internally when
# the watchdog notices a goto was accepted but never produced an
# AutoGoto event (silent symptom of below-horizon, stale View state,
# etc).
INFERRED_NO_AUTOGOTO_CODE = -1


@dataclass(frozen=True)
class FirmwareError:
    """Snapshot of a firmware-side error event the UI should surface."""

    when_utc: datetime
    event_name: str          # 'ScopeTrack', 'ScopeGoto', etc, or 'goto_watchdog'
    code: int                # firmware error code, or INFERRED_NO_AUTOGOTO_CODE
    error: str               # firmware error string ('below horizon', ...)
    state: str               # 'fail', 'off', etc.
    message: str             # operator-friendly explanation
    target_name: Optional[str] = None
    raw: dict = field(default_factory=dict)


class FirmwareErrorMonitor:
    """Tracks the most recent user-actionable firmware error.

    Single instance lives for the process lifetime, owned by
    :mod:`root_app`/:mod:`device.live_tracker_service` analog (created
    when the first ``Seestar`` device boots, similar to the sun monitor).

    Thread-safety: all reads/writes go through an internal lock. The
    front-end polls :meth:`get_pending` every few seconds.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_error: Optional[FirmwareError] = None
        # Per-target watchdog state for the inferred-error path.
        # Key: target_name; value: (submit_time, watchdog Timer or None).
        # Cleared when AutoGoto state="working" arrives for the same target.
        self._pending_goto_target: Optional[str] = None
        self._pending_goto_submit: Optional[datetime] = None
        self._pending_goto_autogoto_fired = False
        self._watchdog_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------
    # Event-loop facing API
    # ------------------------------------------------------------

    def observe(self, parsed_data: dict) -> None:
        """Inspect one firmware event. Records an error if applicable.

        Safe to call on every event; cheap when not interesting.
        """
        event_name = parsed_data.get("Event")
        if event_name not in _WATCHED_EVENT_NAMES:
            # Outside our watch set; but we still want to watch
            # AutoGoto state changes so the watchdog can clear itself.
            return

        # Watchdog clearance: any AutoGoto event with state
        # 'working' / 'start' means the firmware DID engage AutoGoto.
        if event_name == "AutoGoto":
            state = parsed_data.get("state")
            if state in ("working", "start", "complete"):
                with self._lock:
                    self._pending_goto_autogoto_fired = True

        # Direct-error detection
        code = parsed_data.get("code")
        state = parsed_data.get("state")
        error = parsed_data.get("error")
        if (
            code in USER_ACTIONABLE_CODES
            and state in ("fail", "off")
            and error
        ):
            self._record(FirmwareError(
                when_utc=datetime.now(timezone.utc),
                event_name=event_name,
                code=int(code),
                error=str(error),
                state=str(state),
                message=USER_ACTIONABLE_CODES.get(int(code), str(error)),
                target_name=parsed_data.get("target_name")
                or (self._pending_goto_target if self._pending_goto_target else None),
                raw=parsed_data,
            ))

    def notify_goto_submitted(
        self,
        target_name: str,
        watchdog_seconds: float = 6.0,
        on_silent_fail: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Record that a goto was just submitted. Starts a watchdog timer.

        If no AutoGoto event with a 'working'/'start' state is seen
        within ``watchdog_seconds``, an inferred error is recorded
        (covers the silent-below-horizon symptom). If ``on_silent_fail``
        is supplied, it is called with the target name as the only
        argument so the caller can clear its own goto-in-progress
        state (otherwise the next goto would be rejected with "mount
        is in goto routine").
        """
        with self._lock:
            self._pending_goto_target = target_name
            self._pending_goto_submit = datetime.now(timezone.utc)
            self._pending_goto_autogoto_fired = False
            # Cancel any prior pending watchdog (we ignore the result;
            # daemon threads die when the process exits).
            t = threading.Thread(
                target=self._watchdog_run,
                args=(target_name, watchdog_seconds, on_silent_fail),
                daemon=True,
                name=f"firmware-err-watchdog:{target_name}",
            )
            self._watchdog_thread = t
        t.start()

    def _watchdog_run(
        self,
        target_name: str,
        watchdog_seconds: float,
        on_silent_fail: Optional[Callable[[str], None]] = None,
    ) -> None:
        import time as _time
        _time.sleep(watchdog_seconds)
        fire = False
        with self._lock:
            # Only fire inferred error if THIS submit is still the
            # pending one AND AutoGoto never came back.
            if (
                self._pending_goto_target == target_name
                and not self._pending_goto_autogoto_fired
            ):
                inferred = FirmwareError(
                    when_utc=datetime.now(timezone.utc),
                    event_name="goto_watchdog",
                    code=INFERRED_NO_AUTOGOTO_CODE,
                    error="Goto didn't start",
                    state="inferred",
                    message=(
                        f"'{target_name}' accepted but mount never moved. "
                        "Most likely target is below horizon."
                    ),
                    target_name=target_name,
                    raw={},
                )
                self._last_error_unlocked(inferred)
                fire = True
        # Call the callback OUTSIDE the lock to avoid holding it during
        # potentially-slow caller code (e.g. logging, firmware events).
        if fire and on_silent_fail is not None:
            try:
                on_silent_fail(target_name)
            except Exception:
                # Don't let caller exceptions kill the watchdog thread.
                pass

    # ------------------------------------------------------------
    # Front-end facing API
    # ------------------------------------------------------------

    def get_pending(self) -> Optional[FirmwareError]:
        with self._lock:
            return self._last_error

    def dismiss(self) -> None:
        with self._lock:
            self._last_error = None

    def clear_pending_goto(self) -> None:
        """Called when force_stop_goto runs — drops the watchdog target.

        Avoids the watchdog firing after the operator has explicitly
        cancelled the goto.
        """
        with self._lock:
            self._pending_goto_target = None

    # ------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------

    def _record(self, error: FirmwareError) -> None:
        with self._lock:
            self._last_error_unlocked(error)

    def _last_error_unlocked(self, error: FirmwareError) -> None:
        self._last_error = error
        # The pending goto watchdog should not fire again for this
        # target — clear it.
        self._pending_goto_target = None


# Module-level singleton. Constructed eagerly on first import so callers
# can use get_firmware_error_monitor() without worrying about boot
# ordering. The constructor is cheap (no threads, no I/O); the watchdog
# thread is only spawned when a goto is actually submitted.
_monitor: FirmwareErrorMonitor = FirmwareErrorMonitor()


def get_firmware_error_monitor() -> FirmwareErrorMonitor:
    return _monitor
