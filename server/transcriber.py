"""
Ghost Sync — Whisper Transcriber
==================================
Generates subtitle (SRT) files from audio using Faster-Whisper.
Runs locally on CPU — no API key needed.

Model: 'base' (150MB download, ~1GB RAM, ~10 min for a 2hr movie on CPU)
"""

import os
import tempfile
from faster_whisper import WhisperModel


# Default model — 'base' is the sweet spot for CPU
MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")

# Cache the model so it only loads once
_model = None


def get_model():
    """Load the Whisper model (downloads on first use, ~150MB)."""
    global _model
    if _model is None:
        print(f"[Whisper] Loading model '{MODEL_SIZE}'...")
        print(f"          (First time will download ~150MB, then it's cached)")
        _model = WhisperModel(
            MODEL_SIZE,
            device="cpu",
            compute_type="int8",  # fastest on CPU, lower memory
        )
        print(f"[Whisper] Model loaded!")
    return _model


def transcribe_audio(audio_path, language=None):
    """
    Transcribe an audio file to a list of segments.

    Args:
        audio_path: path to audio file (WAV, MP3, etc.)
        language: language code (e.g., 'en'). None = auto-detect.

    Returns list of dicts:
        [{"start": 0.0, "end": 2.5, "text": "Hello world"}, ...]
    """
    model = get_model()

    print(f"[Whisper] Transcribing: {os.path.basename(audio_path)}")
    print(f"          This may take a few minutes on CPU...")

    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5,
        vad_filter=True,       # skip silence (faster!)
        vad_parameters=dict(
            min_silence_duration_ms=500,
        ),
        hallucination_silence_threshold=2.0,  # skip hallucinated text in silence
    )

    import time as _time

    detected_lang = info.language
    duration = info.duration or 0
    duration_str = f"{int(duration//60):02d}:{int(duration%60):02d}" if duration else "??:??"
    print(f"[Whisper] Detected language: {detected_lang} ({info.language_probability:.0%})")
    print(f"[Whisper] Audio duration: {duration_str}")
    print(f"[Whisper] Transcribing... (live progress below)")
    print(f"{'─' * 60}")

    results = []
    prev_text = ""
    start_time = _time.time()
    for segment in segments:
        text = _clean_text(segment.text)

        # Skip empty, too-short, or repeated segments
        if not text or len(text) < 2:
            continue
        if text == prev_text:  # skip exact duplicates
            continue

        results.append({
            "start": segment.start,
            "end": segment.end,
            "text": text,
        })
        prev_text = text

        # Progress output
        pos = segment.end
        pos_str = f"{int(pos//60):02d}:{int(pos%60):02d}"
        pct = f" ({pos/duration*100:.0f}%)" if duration > 0 else ""
        elapsed = _time.time() - start_time
        preview = text[:50] + "..." if len(text) > 50 else text
        print(f"  [{pos_str}/{duration_str}]{pct} seg {len(results)} | \"{preview}\"")

    elapsed_total = _time.time() - start_time
    print(f"{'─' * 60}")
    print(f"[Whisper] Done! {len(results)} segments in {elapsed_total:.0f}s")
    return results, detected_lang


def _clean_text(text):
    """Remove NUL bytes, non-printable chars, and clean up whitespace."""
    import re
    # Remove NUL bytes and other control characters
    text = text.replace('\x00', '')
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def segments_to_srt(segments):
    """
    Convert transcription segments to SRT format string.

    Input: [{"start": 0.0, "end": 2.5, "text": "Hello"}, ...]
    Output: SRT formatted string
    """
    lines = []

    for i, seg in enumerate(segments, 1):
        start = _format_timestamp(seg["start"])
        end = _format_timestamp(seg["end"])
        text = seg["text"]

        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")  # blank line between entries

    return "\n".join(lines)


def _format_timestamp(seconds):
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def transcribe_video(video_path, language=None):
    """
    Full pipeline: transcribe video → return SRT string.
    Whisper can handle video files directly (uses ffmpeg internally).

    Returns:
        dict with keys: srt_content, language, segment_count
    """
    print(f"[Whisper] Transcribing video: {os.path.basename(video_path)}")

    # Whisper/faster-whisper can read video files directly via ffmpeg
    # No need for manual audio extraction — avoids NUL byte issues
    segments, detected_lang = transcribe_audio(video_path, language=language)

    # Convert to SRT
    srt_content = segments_to_srt(segments)

    return {
        "srt_content": srt_content,
        "language": detected_lang,
        "segment_count": len(segments),
    }


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m server.transcriber video.mp4")
        print("       python -m server.transcriber audio.wav")
        sys.exit(1)

    filepath = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else None

    if filepath.endswith((".mp4", ".mkv", ".avi", ".mov", ".wmv")):
        result = transcribe_video(filepath, language=lang)
    else:
        segments, detected = transcribe_audio(filepath, language=lang)
        result = {
            "srt_content": segments_to_srt(segments),
            "language": detected,
            "segment_count": len(segments),
        }

    print(f"\n{'='*60}")
    print(f"Language: {result['language']}")
    print(f"Segments: {result['segment_count']}")
    print(f"{'='*60}")
    print(result["srt_content"][:500])  # first 500 chars
    print("...")
