"""
Ghost Sync — Sync Engine (reusable library)
============================================
Extracted from sync_test.py for server-side use.
Core algorithm: FFT cross-correlation between VAD signal and SRT presence signal.
"""

import os
import subprocess
import tempfile

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


# =============================================================================
# Audio Extraction
# =============================================================================

def extract_audio_pcm(video_path):
    """
    Extract audio from a video file using ffmpeg, piped directly to memory.
    No temp file needed — saves disk I/O and avoids 70MB+ WAV files.

    Returns: raw PCM bytes (16kHz, mono, 16-bit signed little-endian)
    """
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",                  # no video
        "-acodec", "pcm_s16le", # 16-bit PCM
        "-ac", "1",             # mono
        "-ar", str(SAMPLE_RATE),# 16kHz
        "-f", "s16le",          # raw PCM output (no WAV header)
        "pipe:1",               # output to stdout
        "-loglevel", "quiet"
    ]

    result = subprocess.run(cmd, capture_output=True, check=True)
    audio_bytes = result.stdout

    duration_sec = len(audio_bytes) / (SAMPLE_RATE * 2)
    size_mb = len(audio_bytes) / (1024 * 1024)
    print(f"  Audio extracted: {duration_sec:.1f}s ({size_mb:.1f} MB in memory, no temp file)")

    return audio_bytes


def extract_audio_to_opus(video_path, output_path=None):
    """
    Extract audio compressed as Opus 16kbps — for bandwidth-efficient uploads.
    A 39-minute movie goes from ~71MB (WAV) → ~150KB (Opus).

    Per workbook spec:
      Codec: Opus | Bitrate: 16kbps | Mono | 16kHz | VoIP mode
    """
    if output_path is None:
        output_path = os.path.join(
            tempfile.gettempdir(),
            f"ghost_sync_audio_{os.getpid()}.opus"
        )

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",                      # no video
        "-acodec", "libopus",       # Opus codec
        "-b:a", "16k",              # 16kbps bitrate
        "-ac", "1",                 # mono
        "-ar", str(SAMPLE_RATE),    # 16kHz
        "-application", "voip",     # optimized for speech
        output_path,
        "-y",                       # overwrite
        "-loglevel", "quiet"
    ]

    subprocess.run(cmd, check=True)

    file_size = os.path.getsize(output_path)
    print(f"  Opus audio: {file_size / 1024:.1f} KB (for upload)")

    return output_path


# =============================================================================
# VAD Signal
# =============================================================================

def audio_to_vad_signal(audio_bytes):
    """
    Run Voice Activity Detection on raw PCM audio.
    Returns a binary numpy array: 1 = speech, 0 = silence.
    """
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    frame_length = int(SAMPLE_RATE * VAD_FRAME_MS / 1000) * 2  # bytes per frame

    vad_signal = []
    for i in range(0, len(audio_bytes) - frame_length, frame_length):
        frame = audio_bytes[i:i + frame_length]
        try:
            is_speech = vad.is_speech(frame, SAMPLE_RATE)
        except Exception:
            is_speech = False
        vad_signal.append(1 if is_speech else 0)

    return np.array(vad_signal, dtype=np.float64)


# =============================================================================
# SRT Presence Signal
# =============================================================================

def srt_to_presence_signal(srt_path, total_frames, shift_seconds=0.0):
    """
    Convert an SRT file to a binary presence signal.
    1 = subtitle displayed, 0 = no subtitle.
    """
    subs = pysrt.open(srt_path)
    signal = np.zeros(total_frames, dtype=np.float64)

    shift_ms = shift_seconds * 1000

    for sub in subs:
        start_ms = sub.start.ordinal + shift_ms + SRT_ONSET_CORRECTION_MS
        end_ms = sub.end.ordinal + shift_ms + SRT_ONSET_CORRECTION_MS

        start_frame = max(0, int(start_ms / VAD_FRAME_MS))
        end_frame = min(total_frames, int(end_ms / VAD_FRAME_MS))

        if start_frame < end_frame:
            signal[start_frame:end_frame] = 1

    return signal


# =============================================================================
# FFT Cross-Correlation
# =============================================================================

def find_offset(vad_signal, srt_signal):
    """
    FFT cross-correlation to find the time offset between two signals.

    Returns:
        offset_seconds (float): detected offset
        confidence (float): peak-to-mean ratio
    """
    # Smooth both signals
    vad_smooth = gaussian_filter1d(vad_signal, sigma=SMOOTHING_SIGMA)
    srt_smooth = gaussian_filter1d(srt_signal, sigma=SMOOTHING_SIGMA)

    # Pad to prevent circular artifacts
    n = len(vad_smooth) + len(srt_smooth)

    # Cross-correlation in frequency domain
    fft_vad = np.fft.rfft(vad_smooth, n=n)
    fft_srt = np.fft.rfft(srt_smooth, n=n)
    correlation = np.fft.irfft(fft_vad * np.conj(fft_srt))

    # Peak = offset in frames
    peak_index = np.argmax(correlation)
    if peak_index > n // 2:
        peak_index -= n

    offset_seconds = peak_index * VAD_FRAME_MS / 1000

    # Confidence
    peak_value = np.max(correlation)
    mean_value = np.mean(np.abs(correlation))
    confidence = peak_value / mean_value if mean_value > 0 else 0

    return offset_seconds, confidence


# =============================================================================
# High-Level API (what the server calls)
# =============================================================================

def sync_subtitle(video_path, srt_path):
    """
    Full sync pipeline: extract audio → VAD → SRT signal → FFT correlation.

    Returns dict with:
        offset (float): seconds to shift SRT for correct sync
        confidence (float): how reliable the result is
    """
    # Extract audio directly to memory (no temp file!)
    audio_bytes = extract_audio_pcm(video_path)

    # Build signals
    vad_signal = audio_to_vad_signal(audio_bytes)
    srt_signal = srt_to_presence_signal(srt_path, total_frames=len(vad_signal))

    # Find offset
    offset, confidence = find_offset(vad_signal, srt_signal)

    return {
        "offset": offset,
        "confidence": confidence,
    }
