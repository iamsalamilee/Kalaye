"""
Kalaye — Real-Time Shazam Sync Engine
======================================
Captures system audio every 10-15 seconds via WASAPI loopback,
fingerprints it, and corrects the subtitle clock to the exact
movie position. Runs 100% locally.

Features:
  - Dual-layer matching (constellation + mel verification)
  - Speed-aware: compensates for playback speed changes
  - Hysteresis: smooths corrections to prevent jitter
  - Seek detection: instantly corrects large jumps
  - Pause detection: saves CPU during silence

Usage:
    from client.spectrogram_sync import load_db
    from client.realtime_sync import ShazamSyncEngine

    db = load_db("movie.fpdb")
    engine = ShazamSyncEngine(db)
    engine.position_found.connect(player.sync_to_position)
    engine.start()
"""

import time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from client.spectrogram_sync import find_position


# ============================================================================
# Configuration
# ============================================================================

CAPTURE_SECONDS = 5          # duration of each audio capture
CYCLE_SECONDS = 12           # seconds between sync checks
SAMPLE_RATE = 16000          # must match spectrogram_sync.py

# Hysteresis: prevent jitter from one-off bad matches
HYSTERESIS_MS = 100          # ignore corrections smaller than this
MAX_HISTORY = 5              # number of recent corrections to average

# Seek detection: if position jumps more than this, apply immediately
SEEK_THRESHOLD_MS = 10000    # 10 seconds = user probably seeked

# Silence detection: skip matching if audio is too quiet
SILENCE_RMS_THRESHOLD = 0.005  # below this = silence (float64 scale)

# Confidence tiers
HIGH_CONFIDENCE = 40         # apply immediately, no verification needed
MEDIUM_CONFIDENCE = 20       # apply with hysteresis smoothing
LOW_CONFIDENCE = 10          # too weak, skip this cycle


class ShazamSyncEngine(QThread):
    """Continuously syncs subtitles using spectrogram fingerprint matching.

    Emits position_found with the exact movie position every time
    a confident match is made. Connect this to SubtitlePlayer.sync_to_position.

    Signals:
        position_found(float, float):
            (exact_movie_position_ms, confidence)
            confidence = number of agreeing hashes

        sync_status(str):
            Human-readable status for UI display

        sync_ready(bool):
            True when the engine is initialized and listening
    """

    position_found = pyqtSignal(float, float)   # (position_ms, confidence)
    sync_status = pyqtSignal(str)                # status message for overlay
    sync_ready = pyqtSignal(bool)                # ready to sync

    def __init__(self, fingerprint_db, playback_speed=1.0):
        """Initialize the sync engine.

        Args:
            fingerprint_db: dict from build_fingerprint_db() or load_db().
            playback_speed: initial playback speed (1.0 = normal).
        """
        super().__init__()
        self.fingerprint_db = fingerprint_db
        self.playback_speed = playback_speed
        self.running = True

        # Correction history for hysteresis smoothing
        self._correction_history = []

        # Last known position for seek detection
        self._last_position_ms = None

        # Stats for logging
        self._match_count = 0
        self._skip_count = 0

    def set_speed(self, speed):
        """Update the playback speed (called when user changes speed).

        Args:
            speed: new playback speed (e.g., 1.0, 1.5, 2.0).
        """
        self.playback_speed = speed

    def run(self):
        """Main sync loop. Runs in a background QThread."""
        try:
            import soundcard
        except ImportError:
            self.sync_status.emit("⚠️ Install 'soundcard': pip install soundcard")
            print("[ShazamSync] ERROR: 'soundcard' not installed. "
                  "Run: pip install soundcard")
            return

        self.sync_status.emit("🔊 Initializing audio sync...")

        try:
            # Get the loopback microphone for the default speaker
            # (loopback = captures system audio output, not physical mic)
            default_speaker = soundcard.default_speaker()
            loopback = None

            for mic in soundcard.all_microphones(include_loopback=True):
                if mic.isloopback and default_speaker.name in mic.name:
                    loopback = mic
                    break

            if loopback is None:
                self.sync_status.emit("⚠️ No loopback device found")
                print("[ShazamSync] ERROR: Could not find loopback mic for default speaker")
                print(f"[ShazamSync] Default speaker: {default_speaker.name}")
                print(f"[ShazamSync] Available loopback mics: "
                      f"{[m.name for m in soundcard.all_microphones(include_loopback=True) if m.isloopback]}")
                return

            print(f"[ShazamSync] Using loopback device: {loopback.name}")

        except Exception as e:
            self.sync_status.emit(f"⚠️ No audio device: {str(e)[:40]}")
            print(f"[ShazamSync] ERROR: Could not get loopback device: {e}")
            return

        self.sync_ready.emit(True)
        self.sync_status.emit("🎯 Listening for audio...")
        print("[ShazamSync] Engine started — listening for audio")

        while self.running:
            try:
                self._sync_cycle(loopback)
            except Exception as e:
                self.sync_status.emit(f"⚠️ {str(e)[:40]}")
                print(f"[ShazamSync] Cycle error: {e}")

            # Wait before next cycle
            # Use short sleeps so we can stop quickly when self.running = False
            for _ in range(int(CYCLE_SECONDS * 10)):
                if not self.running:
                    break
                time.sleep(0.1)

        print("[ShazamSync] Engine stopped")

    def _sync_cycle(self, loopback):
        """One capture → match → correct cycle."""
        import soundcard

        # 1. Capture system audio
        try:
            with loopback.recorder(samplerate=SAMPLE_RATE, channels=1) as mic:
                audio = mic.record(numframes=SAMPLE_RATE * CAPTURE_SECONDS)
        except Exception as e:
            self.sync_status.emit(f"⚠️ Capture failed: {str(e)[:30]}")
            return

        samples = audio.flatten().astype(np.float64)

        # 2. Silence detection — skip if nothing is playing
        rms = np.sqrt(np.mean(samples ** 2))
        if rms < SILENCE_RMS_THRESHOLD:
            self._skip_count += 1
            self.sync_status.emit("⏸ Silence detected — waiting...")
            return

        # 3. Speed compensation — if playing at 1.5x, the captured audio
        #    is time-compressed relative to the original. Resample to 1.0x
        #    so the spectrogram matches the fingerprint DB.
        if self.playback_speed != 1.0 and self.playback_speed > 0:
            original_len = int(len(samples) * self.playback_speed)
            samples = np.interp(
                np.linspace(0, len(samples) - 1, original_len),
                np.arange(len(samples)),
                samples,
            )

        # 4. Layer 1: Constellation hash matching
        position_ms, confidence, total_matches = find_position(
            samples, self.fingerprint_db
        )

        if position_ms is None:
            self._skip_count += 1
            self.sync_status.emit(
                f"⏳ Listening... ({total_matches} partial, "
                f"need {MIN_HASH_MATCHES_LABEL})"
            )
            return

        # 5. Adjust position: add capture duration since audio was captured
        #    over the last CAPTURE_SECONDS. The position corresponds to the
        #    START of the capture, so add the duration to get "now".
        adjusted_ms = position_ms + (CAPTURE_SECONDS * 1000.0 * self.playback_speed)

        # 6. Confidence-based action
        if confidence >= HIGH_CONFIDENCE:
            # High confidence: apply immediately
            self._apply_correction(adjusted_ms, confidence, immediate=True)
        elif confidence >= MEDIUM_CONFIDENCE:
            # Medium confidence: apply with hysteresis smoothing
            self._apply_correction(adjusted_ms, confidence, immediate=False)
        else:
            # Low confidence: log but don't apply
            self._skip_count += 1
            self.sync_status.emit(
                f"⏳ Weak match ({confidence} hashes, need {MEDIUM_CONFIDENCE}+)"
            )

    def _apply_correction(self, position_ms, confidence, immediate=False):
        """Apply a position correction with optional hysteresis.

        Args:
            position_ms: detected movie position in ms.
            confidence: number of agreeing hashes.
            immediate: if True, skip smoothing (for high confidence or seeks).
        """
        # Seek detection: if position jumped massively, apply immediately
        if self._last_position_ms is not None:
            jump = abs(position_ms - self._last_position_ms)
            expected_jump = CYCLE_SECONDS * 1000.0 * self.playback_speed
            if jump > SEEK_THRESHOLD_MS and jump > expected_jump * 3:
                immediate = True
                print(f"[ShazamSync] Seek detected: jumped {jump / 1000:.1f}s")

        if immediate:
            # Apply directly
            self._correction_history.clear()
            self._last_position_ms = position_ms
            self._match_count += 1
            self.position_found.emit(position_ms, float(confidence))
            self.sync_status.emit(f"🎯 Synced! ({confidence} hashes)")
            return

        # Hysteresis: add to history, emit smoothed average
        self._correction_history.append(position_ms)
        if len(self._correction_history) > MAX_HISTORY:
            self._correction_history.pop(0)

        # Need at least 2 data points to smooth
        if len(self._correction_history) < 2:
            self.sync_status.emit(
                f"🔄 Confirming... ({len(self._correction_history)}/{2})"
            )
            return

        # Smoothed position = median of recent corrections
        # (Median is robust against one-off outliers)
        smoothed_ms = float(np.median(self._correction_history))

        # Only emit if the correction is meaningful (> hysteresis threshold)
        if self._last_position_ms is not None:
            correction = abs(smoothed_ms - self._last_position_ms)
            expected = CYCLE_SECONDS * 1000.0 * self.playback_speed
            drift = abs(correction - expected)

            if drift < HYSTERESIS_MS:
                # Drift is tiny — subtitles are already synced, no action needed
                self.sync_status.emit(f"✅ In sync (drift: {drift:.0f}ms)")
                self._last_position_ms = smoothed_ms
                return

        self._last_position_ms = smoothed_ms
        self._match_count += 1
        self.position_found.emit(smoothed_ms, float(confidence))
        self.sync_status.emit(f"🎯 Synced! ({confidence} hashes, smoothed)")

    def stop(self):
        """Stop the sync engine."""
        self.running = False

    def get_stats(self):
        """Return sync statistics for debugging.

        Returns:
            dict with match_count, skip_count, history_size.
        """
        return {
            'match_count': self._match_count,
            'skip_count': self._skip_count,
            'history_size': len(self._correction_history),
        }


# Label for status messages (avoids importing the constant name)
MIN_HASH_MATCHES_LABEL = MEDIUM_CONFIDENCE
