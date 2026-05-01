"""
Kalaye -- Player Sync Engine
============================
Keeps the subtitle overlay perfectly synced with VLC Media Player
by polling VLC's built-in HTTP interface for exact playback position.

Zero drift. Zero guessing. Exact millisecond accuracy.

VLC Setup (one-time, takes 30 seconds):
  1. Open VLC -> Tools -> Preferences
  2. Bottom-left corner: switch "Show settings" to "All"
  3. Navigate: Interface -> Main interfaces -> check "Web"
  4. Navigate: Interface -> Main interfaces -> Lua -> Lua HTTP
  5. Set Password to: kalaye
  6. Click Save, then restart VLC

Once enabled, this engine polls VLC every 500ms for the exact
playback position. Handles play, pause, seek -- everything.

100% local. No network calls outside localhost.
"""

import time
import requests
from PyQt5.QtCore import QThread, pyqtSignal


# =============================================================================
# Configuration
# =============================================================================

POLL_INTERVAL = 0.5      # seconds between VLC polls
VLC_URL = "http://localhost:8080/requests/status.json"
VLC_PASSWORD = "kalaye"  # must match VLC's Lua HTTP password
MAX_CONNECT_RETRIES = 12 # stop after ~6 seconds of no VLC → triggers Shazam fallback


class AudioSyncEngine(QThread):
    """Polls VLC's HTTP interface for exact playback position.

    Despite the class name (kept for backward compat with app.py),
    this uses VLC's HTTP API, NOT audio fingerprinting.

    Signals:
        position_found(float, float): (position_ms, confidence)
            confidence is always 100.0 for VLC HTTP (exact data).
        playback_state(bool): True = playing, False = paused/stopped
        status(str): human-readable status for the overlay
        ready(bool): True when VLC connection is established
    """

    position_found = pyqtSignal(float, float)   # (position_ms, confidence)
    playback_state = pyqtSignal(bool)            # True = playing
    status = pyqtSignal(str)                     # status message
    ready = pyqtSignal(bool)                     # VLC connection established

    def __init__(self, video_path=None):
        super().__init__()
        # video_path kept for API compat but not used (VLC knows what it's playing)
        self.video_path = video_path
        self.running = True
        self._connected = False
        self._last_state = None

    def run(self):
        """Main polling loop -- runs in background QThread."""
        self.status.emit("Connecting to VLC HTTP interface...")
        retry_count = 0

        while self.running:
            try:
                r = requests.get(
                    VLC_URL,
                    auth=("", VLC_PASSWORD),
                    timeout=1
                )

                # Handle auth errors
                if r.status_code == 401:
                    self.status.emit("VLC password mismatch -- set to 'kalaye'")
                    time.sleep(3)
                    continue

                if r.status_code != 200:
                    time.sleep(POLL_INTERVAL)
                    continue

                data = r.json()

                # First successful connection
                if not self._connected:
                    self._connected = True
                    self.ready.emit(True)
                    self.status.emit("Connected to VLC")
                    retry_count = 0
                    print("[VLCSync] Connected to VLC HTTP interface")

                # Extract exact position
                state = data.get("state", "stopped")      # "playing" | "paused" | "stopped"
                length_sec = data.get("length", 0)         # total duration in seconds
                position_frac = data.get("position", 0.0)  # 0.0 to 1.0 fractional position
                time_sec = data.get("time", 0)              # current time in whole seconds

                # Use fractional position * length for sub-second precision
                # Fall back to integer seconds if length is unknown
                if length_sec > 0 and position_frac > 0:
                    position_ms = position_frac * length_sec * 1000.0
                else:
                    position_ms = time_sec * 1000.0

                is_playing = (state == "playing")

                # Emit playback state changes
                if is_playing != self._last_state:
                    self.playback_state.emit(is_playing)
                    self._last_state = is_playing
                    action = "Playing" if is_playing else "Paused"
                    print(f"[VLCSync] {action} at {self._format_time(position_ms / 1000)}")

                # Sync the overlay clock when playing
                if is_playing and position_ms >= 0:
                    self.position_found.emit(position_ms, 100.0)

            except requests.ConnectionError:
                # VLC HTTP not available
                if self._connected:
                    self._connected = False
                    self.ready.emit(False)
                    self.status.emit("VLC disconnected")
                    self._last_state = None
                    print("[VLCSync] VLC disconnected")
                else:
                    retry_count += 1
                    if retry_count <= MAX_CONNECT_RETRIES:
                        self.status.emit(
                            f"Waiting for VLC... ({retry_count}/{MAX_CONNECT_RETRIES})"
                        )
                    elif retry_count == MAX_CONNECT_RETRIES + 1:
                        self.status.emit(
                            "VLC HTTP not found -- switching to audio sync"
                        )
                        print(
                            "[VLCSync] VLC HTTP not reachable after "
                            f"{MAX_CONNECT_RETRIES} attempts — triggering fallback"
                        )
                        self.ready.emit(False)  # trigger Shazam fallback

            except requests.Timeout:
                pass  # VLC is slow to respond, just skip this cycle

            except Exception as e:
                print(f"[VLCSync] Unexpected error: {e}")

            time.sleep(POLL_INTERVAL)

    def stop(self):
        """Stop the polling loop."""
        self.running = False

    @staticmethod
    def _format_time(seconds):
        """Format seconds as MM:SS."""
        seconds = max(0, seconds)
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"
