"""
Ghost Sync — Phase 1: Sync Test Harness
========================================
Proves that FFT cross-correlation between a VAD audio signal and an SRT
subtitle presence signal can recover the correct time offset without
any speech recognition.

Usage:
    python sync_test.py <video_file> <srt_file> [--shift SECONDS]

    --shift SECONDS   Deliberately shift the SRT by this many seconds
                      before running sync, so you know the correct answer.
                      If omitted, the script just reports the detected offset.

Examples:
    # Test with a known 5-second shift
    python sync_test.py movie.mkv movie.srt --shift 5.0

    # Just detect the current offset
    python sync_test.py movie.mkv movie.srt
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pysrt
import webrtcvad
from scipy.ndimage import gaussian_filter1d


# =============================================================================
# CONFIGURATION
# =============================================================================

SAMPLE_RATE = 16000          # Hz — matches webrtcvad requirement
VAD_AGGRESSIVENESS = 2       # 0-3, 2 is balanced
VAD_FRAME_MS = 30            # milliseconds per VAD frame
SMOOTHING_SIGMA = 20         # Gaussian kernel width for signal smoothing
SRT_ONSET_CORRECTION_MS = -100  # SRT text appears ~100ms after speech starts
PASS_THRESHOLD_SEC = 0.5     # Maximum acceptable offset error


# =============================================================================
# STEP 1 — Extract audio from video file
# =============================================================================

def extract_audio(video_path):
    """
    Extract audio from a video file using ffmpeg.
    Pipes PCM directly to memory — no temp files, no disk I/O.
    Returns: raw PCM bytes (16kHz, mono, 16-bit signed little-endian)
    """
    print(f"[Step 1] Extracting audio from: {os.path.basename(video_path)}")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",                  # no video
        "-acodec", "pcm_s16le", # 16-bit PCM
        "-ac", "1",             # mono
        "-ar", str(SAMPLE_RATE),# 16kHz
        "-f", "s16le",          # raw PCM (no WAV header)
        "pipe:1",               # pipe to stdout
        "-loglevel", "quiet"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError:
        print("\n[ERROR] ffmpeg not found!")
        print("Install ffmpeg: https://ffmpeg.org/download.html")
        print("On Windows: winget install ffmpeg")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] ffmpeg failed: {e}")
        sys.exit(1)

    audio_bytes = result.stdout
    pcm_size_mb = len(audio_bytes) / (1024 * 1024)
    duration_sec = len(audio_bytes) / (SAMPLE_RATE * 2)  # 2 bytes per sample

    # What it WOULD be as Opus (16kbps) — for comparison
    opus_estimated_kb = (duration_sec * 16000 / 8) / 1024  # 16kbps in KB

    print(f"         Duration: {duration_sec:.1f} seconds")
    print(f"         PCM in memory: {pcm_size_mb:.1f} MB (no file written to disk!)")
    print(f"         Opus equivalent: ~{opus_estimated_kb:.0f} KB (for server upload)")

    return audio_bytes


# =============================================================================
# STEP 2 — Run VAD to get speech/silence binary signal
# =============================================================================

def audio_to_vad_signal(audio_bytes):
    """
    Run Voice Activity Detection on raw PCM audio.
    Returns a binary numpy array: 1 = speech, 0 = silence.
    Each element represents one VAD_FRAME_MS frame.
    """
    print("[Step 2] Running Voice Activity Detection...")

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    frame_length = int(SAMPLE_RATE * VAD_FRAME_MS / 1000) * 2  # bytes per frame

    vad_signal = []
    total_frames = 0
    speech_frames = 0

    for i in range(0, len(audio_bytes) - frame_length, frame_length):
        frame = audio_bytes[i:i + frame_length]
        try:
            is_speech = vad.is_speech(frame, SAMPLE_RATE)
        except Exception:
            is_speech = False

        vad_signal.append(1 if is_speech else 0)
        total_frames += 1
        if is_speech:
            speech_frames += 1

    signal = np.array(vad_signal, dtype=np.float64)
    speech_pct = (speech_frames / total_frames * 100) if total_frames > 0 else 0
    duration_sec = total_frames * VAD_FRAME_MS / 1000

    print(f"         {total_frames} frames ({duration_sec:.1f}s), "
          f"{speech_pct:.1f}% speech detected")

    return signal


# =============================================================================
# STEP 3 — Convert SRT to a matching binary presence signal
# =============================================================================

def srt_to_presence_signal(srt_path, total_frames, shift_seconds=0.0):
    """
    Convert an SRT file to a binary presence signal.
    1 = subtitle text is displayed, 0 = no subtitle.
    Each element represents one VAD_FRAME_MS frame.

    shift_seconds: deliberately shift all SRT timestamps by this amount
                   (positive = delay subtitles, negative = advance them)
    """
    print("[Step 3] Converting SRT to presence signal...")

    subs = pysrt.open(srt_path)
    signal = np.zeros(total_frames, dtype=np.float64)

    # Apply SRT onset correction: subtitles appear ~100ms after speech starts
    correction_ms = SRT_ONSET_CORRECTION_MS
    shift_ms = shift_seconds * 1000

    subtitle_count = 0
    for sub in subs:
        # Get start/end in milliseconds, apply shift and correction
        start_ms = sub.start.ordinal + shift_ms + correction_ms
        end_ms = sub.end.ordinal + shift_ms + correction_ms

        # Convert to frame indices
        start_frame = int(start_ms / VAD_FRAME_MS)
        end_frame = int(end_ms / VAD_FRAME_MS)

        # Clamp to valid range
        start_frame = max(0, start_frame)
        end_frame = min(total_frames, end_frame)

        if start_frame < end_frame:
            signal[start_frame:end_frame] = 1
            subtitle_count += 1

    coverage = np.sum(signal) / total_frames * 100 if total_frames > 0 else 0
    print(f"         {subtitle_count} subtitle blocks, {coverage:.1f}% signal coverage")

    if shift_seconds != 0:
        print(f"         Applied deliberate shift: {shift_seconds:+.1f}s")

    return signal


# =============================================================================
# STEP 4 — Smooth both signals
# =============================================================================

def smooth_signal(signal, sigma=SMOOTHING_SIGMA):
    """
    Apply Gaussian smoothing to a binary signal.
    This fills small gaps in VAD (brief silence mid-sentence)
    and makes the correlation more robust.
    """
    return gaussian_filter1d(signal, sigma=sigma)


# =============================================================================
# STEP 5 — FFT Cross-Correlation to find the offset
# =============================================================================

def find_sync_offset(vad_smooth, srt_smooth):
    """
    Use FFT cross-correlation to find the time offset between
    the VAD signal and the SRT presence signal.

    Returns:
        offset_seconds (float): detected offset in seconds
        confidence (float): peak-to-mean ratio (higher = more confident)
    """
    print("[Step 5] Running FFT cross-correlation...")

    # Pad to prevent circular correlation artifacts
    n = len(vad_smooth) + len(srt_smooth)

    # Transform to frequency domain
    fft_vad = np.fft.rfft(vad_smooth, n=n)
    fft_srt = np.fft.rfft(srt_smooth, n=n)

    # Cross-correlation in frequency domain
    correlation = np.fft.irfft(fft_vad * np.conj(fft_srt))

    # Peak position = offset in frames
    peak_index = np.argmax(correlation)

    # Handle wrap-around: if peak is in the second half, it's a negative offset
    if peak_index > n // 2:
        peak_index -= n

    # Convert frames to seconds
    offset_seconds = peak_index * VAD_FRAME_MS / 1000

    # Confidence: how sharp is the peak vs background noise
    peak_value = np.max(correlation)
    mean_value = np.mean(np.abs(correlation))
    confidence = peak_value / mean_value if mean_value > 0 else 0

    print(f"         Peak at frame {peak_index}, confidence: {confidence:.2f}")

    return offset_seconds, confidence


# =============================================================================
# FPS Mismatch Detection (run after multiple test windows)
# =============================================================================

def detect_fps_mismatch(drift_measurements):
    """
    Detect linear drift indicating an FPS mismatch between video and SRT.

    drift_measurements: list of (elapsed_seconds, observed_drift_seconds)

    Returns: correction_factor to multiply timestamps by.
    A 25fps SRT on 23.976fps video drifts ~0.1s per 60s.
    """
    if len(drift_measurements) < 3:
        return 1.0  # not enough data

    times = np.array([m[0] for m in drift_measurements])
    drifts = np.array([m[1] for m in drift_measurements])

    # Fit a line: drift = slope * time + intercept
    slope = np.polyfit(times, drifts, deg=1)[0]

    # If slope is significant, there's an FPS mismatch
    correction_factor = 1.0 + slope

    if abs(slope) > 0.0005:  # more than 0.5ms per second of drift
        print(f"\n[!] Possible FPS mismatch detected!")
        print(f"    Drift rate: {slope * 1000:.2f} ms/second")
        print(f"    Over 1 hour, this would drift: {slope * 3600:.1f} seconds")
        print(f"    Correction factor: {correction_factor:.6f}")

    return correction_factor


# =============================================================================
# MAIN — Run the full sync test
# =============================================================================

def run_sync_test(video_path, srt_path, shift_seconds=0.0):
    """
    Run the complete sync test pipeline:
    1. Extract audio from video
    2. Run VAD to get speech signal
    3. Convert SRT to presence signal (with optional shift)
    4. Smooth both signals
    5. FFT cross-correlation to find offset
    """
    print("=" * 60)
    print("  GHOST SYNC — Phase 1 Sync Test Harness")
    print("=" * 60)
    print(f"\n  Video: {os.path.basename(video_path)}")
    print(f"  SRT:   {os.path.basename(srt_path)}")
    if shift_seconds != 0:
        print(f"  Known shift applied: {shift_seconds:+.1f}s")
    print()

    start_time = time.time()

    # Step 1 — Extract audio (piped to memory, no temp file)
    audio_bytes = extract_audio(video_path)

    # Step 2 — VAD
    vad_signal = audio_to_vad_signal(audio_bytes)

    # Step 3 — SRT presence signal (with deliberate shift)
    srt_signal = srt_to_presence_signal(
        srt_path,
        total_frames=len(vad_signal),
        shift_seconds=shift_seconds
    )

    # Step 4 — Smooth both signals
    print(f"[Step 4] Smoothing signals (Gaussian, sigma={SMOOTHING_SIGMA})...")
    vad_smooth = smooth_signal(vad_signal)
    srt_smooth = smooth_signal(srt_signal)

    # Step 5 — FFT cross-correlation
    offset_seconds, confidence = find_sync_offset(vad_smooth, srt_smooth)

    elapsed = time.time() - start_time

    # =================================================================
    # Results
    # =================================================================
    # The detected offset is the correction needed: a +5s shift should yield ~-5s offset
    # So the error is how far |detected + shift| deviates from zero
    error = abs(offset_seconds + shift_seconds) if shift_seconds != 0 else None

    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Detected offset:  {offset_seconds:+.3f} seconds")
    print(f"  Confidence:       {confidence:.2f}")
    print(f"  Processing time:  {elapsed:.1f} seconds")

    if shift_seconds != 0:
        # We know the correct answer — check accuracy
        passed = error <= PASS_THRESHOLD_SEC

        print()
        print(f"  Known shift:      {shift_seconds:+.3f} seconds")
        print(f"  Error:            {error:.3f} seconds")
        print(f"  Threshold:        +/-{PASS_THRESHOLD_SEC} seconds")
        print()

        if passed:
            print("  >>> PASS — Offset recovered within threshold!")
        else:
            print("  >>> FAIL — Offset error exceeds threshold.")
            print()
            print("  Debugging hints:")
            if confidence < 3.0:
                print("  - Low confidence suggests weak correlation.")
                print("    This may be an action-heavy or music-heavy film")
                print("    where VAD can't distinguish speech from noise.")
            if error > 10:
                print("  - Large error may indicate the SRT doesn't match")
                print("    this video. Check release version / rip type.")
    else:
        print()
        print("  No known shift provided — offset is the detected sync point.")
        print("  Use --shift SECONDS to test against a known offset.")

    print()
    print("=" * 60)

    return {
        "offset": offset_seconds,
        "confidence": confidence,
        "error": error,
        "passed": error is not None and error <= PASS_THRESHOLD_SEC,
        "processing_time": elapsed
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ghost Sync — Phase 1 Sync Test Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sync_test.py movie.mkv movie.srt --shift 5.0
  python sync_test.py episode.mp4 episode.srt
        """
    )

    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("srt", help="Path to the SRT subtitle file")
    parser.add_argument(
        "--shift", type=float, default=0.0,
        help="Deliberately shift SRT by this many seconds (for testing)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.video):
        print(f"[ERROR] Video file not found: {args.video}")
        sys.exit(1)

    if not os.path.exists(args.srt):
        print(f"[ERROR] SRT file not found: {args.srt}")
        sys.exit(1)

    # Run the test
    result = run_sync_test(args.video, args.srt, args.shift)

    # Exit code: 0 = pass or no known shift, 1 = fail
    if result.get("passed") is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
