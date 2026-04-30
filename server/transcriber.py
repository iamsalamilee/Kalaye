"""
Kalaye — Cloud Transcriber (Modal GPU) v2
============================================
Sends audio to H100 GPUs on Modal for transcription
using Faster-Whisper large-v3 (the most accurate model).

v2 fixes applied:
  - FIX 1: task="transcribe" forced — prevents silent translation
  - FIX 2: Language-aware VAD parameters for soft-speech languages
  - FIX 3: EBU R128 loudness normalization on audio extraction
  - FIX 4: Music detection via word-probability + repetition signals
  - FIX 5: Higher beam_size (10) for complex/tonal languages
  - FIX 6: Every segment has a "type" field for debugging
  - FIX 7: Extended hallucination patterns, all existing logic preserved

No local CPU processing — everything runs on cloud H100 GPUs.
"""

import os
import subprocess
import tempfile
import modal

# ============================================================================
# Modal App & Cloud Environment
# ============================================================================

app = modal.App("kalaye-whisper")

# Heavy GPU image for Whisper large-v3 — model baked in at build time.
# device='cpu' during build because Modal build-servers have no GPU.
whisper_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04", add_python="3.11"
    )
    .apt_install("ffmpeg")
    .pip_install("faster-whisper")
    .run_commands(
        # Bake model into image so containers start instantly (no download).
        # device='cpu' is required here because Modal build servers lack GPUs.
        "python -c \"from faster_whisper import WhisperModel; "
        "WhisperModel('large-v3', device='cpu')\""
    )
)


# ============================================================================
# Language-Aware Configuration (Fixes 2 & 5)
# ============================================================================

# Languages with aspirated consonants, tonal features, retroflex sounds, or
# schwa deletion that cause quiet phonemes to fall below default VAD threshold.
SOFT_SPEECH_LANGUAGES = {
    "hi", "bn", "pa", "mr", "gu", "ta", "te", "kn", "ml", "ur",  # South Asian
    "ar", "fa",                                                     # Middle Eastern
    "yo", "ig", "sw",                                               # African
    "zh", "ja", "ko", "th", "vi",                                   # East/SE Asian
}


def get_vad_params_for_language(language: str) -> dict:
    """Return VAD parameters tuned for the given language.

    Soft-speech languages get a lower energy threshold (0.3) so quiet
    phonemes like aspirated stops and retroflexes are not dropped.
    All other languages use the standard Silero defaults.

    Args:
        language: ISO 639-1 code, or None for auto-detect.

    Returns:
        dict of VAD parameters for faster-whisper's vad_parameters kwarg.
    """
    if language and language.lower() in SOFT_SPEECH_LANGUAGES:
        return {
            "threshold": 0.3,                  # lower energy gate for soft phonemes
            "min_silence_duration_ms": 200,     # aggressive split — matches translate-mode granularity
            "speech_pad_ms": 300,               # moderate padding keeps word boundaries intact
        }
    # Default: works well for English, Romance, Germanic, Slavic, etc.
    return {
        "threshold": 0.5,
        "min_silence_duration_ms": 300,         # lowered from 500 — transcribe mode merges too much
        "speech_pad_ms": 150,
    }


def get_beam_size_for_language(language: str) -> int:
    """Return beam_size tuned for the given language.

    Complex/tonal/soft-speech languages benefit from 10 beams
    (more decoding candidates) at a ~20% compute cost increase.

    Args:
        language: ISO 639-1 code, or None for auto-detect.

    Returns:
        int beam_size.
    """
    if language and language.lower() in SOFT_SPEECH_LANGUAGES:
        return 10  # more candidates -> better accuracy for complex phonology
    return 5       # standard beam width


# ============================================================================
# FIX 4: Music Detection
# ============================================================================

def classify_segment_type(text: str, segment) -> str:
    """Classify a Whisper segment as 'music', 'speech', or 'uncertain'.

    Detection signals:
      1. Word probability: if segment has word-level timestamps, low
         average probability (<0.6) indicates Whisper is confused —
         singing consistently produces low-confidence output.
      2. Repetition: chorus-like repetition (unique_ratio < 0.5 with
         more than 3 words) is a strong lyric signal.
      3. Duration: very short segments (< 0.8s, <= 2 words) are ambiguous.

    Args:
        text:    The transcribed text string.
        segment: The faster-whisper Segment object (has .words attribute).

    Returns:
        One of: 'music', 'speech', 'uncertain'.
    """
    words = text.split()
    word_count = len(words)

    # Signal 1: Word-level probability (only if word timestamps exist)
    if hasattr(segment, "words") and segment.words:
        probs = [w.probability for w in segment.words if hasattr(w, "probability")]
        if probs:
            avg_prob = sum(probs) / len(probs)
            if avg_prob < 0.6:
                return "music"  # low confidence = melodic input confusing Whisper

    # Signal 2: Repetition ratio (chorus detection)
    if word_count > 3:
        unique_ratio = len(set(w.lower() for w in words)) / word_count
        if unique_ratio < 0.5:
            return "music"  # heavy repetition = likely lyrics

    # Signal 3: Very short + few words = ambiguous
    seg_duration = getattr(segment, "end", 0) - getattr(segment, "start", 0)
    if seg_duration < 0.8 and word_count <= 2:
        return "uncertain"

    return "speech"


# ============================================================================
# Cloud Transcription (H100 GPU, model baked in)
# ============================================================================

@app.cls(
    gpu="A10",
    image=whisper_image,
    timeout=600,                  # 10 min max per call
    scaledown_window=120,         # keep warm for 2 min between requests
)
class CloudTranscriber:
    """Whisper transcription on H100 GPU with all v2 fixes."""

    @modal.enter()
    def load_model(self):
        """Runs ONCE when container starts. All requests reuse this model."""
        from faster_whisper import WhisperModel
        self.model = WhisperModel(
            "large-v3",
            device="cuda",
            compute_type="float16",  # float16 is faster than int8 on GPU
        )
        print("[Whisper] Model loaded on H100 GPU — container is warm")

    @modal.method()
    def transcribe_chunk(
        self,
        audio_bytes: bytes,
        start_offset: float = 0.0,
        language: str = None,
    ):
        """Transcribe one audio chunk and shift all timestamps by start_offset.

        Applies all 7 fixes:
          - task='transcribe' forced (Fix 1)
          - Language-aware VAD params (Fix 2)
          - beam_size per language (Fix 5)
          - Music detection + formatting (Fix 4)
          - Type field on every segment (Fix 6)
          - Extended hallucination patterns (Fix 7)

        Args:
            audio_bytes:  Raw WAV bytes for this chunk.
            start_offset: Where this chunk starts in the full file (seconds).
            language:     ISO 639-1 code to force, or None for auto.

        Returns:
            dict with segments, detected language, and language_probability.
        """
        import tempfile
        import os

        # Edge case 2: skip empty/corrupt audio chunks
        if len(audio_bytes) < 1000:
            print(f"[Whisper] WARNING: Skipping tiny audio chunk ({len(audio_bytes)} bytes)")
            return {
                "segments": [],
                "language": language or "unknown",
                "language_probability": 0.0,
                "duration": 0,
            }

        # Write WAV bytes to temp file for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            # FIX 2: Get language-tuned VAD parameters
            vad_params = get_vad_params_for_language(language)

            # FIX 5: Get language-tuned beam size
            beam_size = get_beam_size_for_language(language)

            segments_iter, info = self.model.transcribe(
                temp_path,
                task="transcribe",                     # FIX 1: NEVER translate
                language=language,
                beam_size=beam_size,                    # FIX 5: 10 for complex, 5 otherwise
                word_timestamps=True,
                condition_on_previous_text=False,       # prevents hallucination carry-over
                vad_filter=True,                        # skip silence and music
                vad_parameters=vad_params,              # FIX 2: language-aware thresholds
                hallucination_silence_threshold=5.0,
            )

            # Known Whisper hallucination strings during silence/music/credits.
            # FIX 7: Extended with music-specific hallucination patterns.
            HALLUCINATION_PATTERNS = [
                "subtitle", "\uc790\ub9c9", "\ud55c\uae00\uc790\ub9c9", "\uc124\uc815\uc5d0\uc11c",
                "\uac10\uc0ac\ud569\ub2c8\ub2e4", "thank you", "thanks for watching",
                "subscribe", "\uad6c\ub3c5", "\uc88b\uc544\uc694",
                "by \ud55c", "by han", "\uc544\uba58", "amen",
                "www.", ".com", "http",
                # Music-specific hallucinations added in v2
                "\u266b", "\ud83c\udfb5", "la la", "na na", "hmm hmm",
            ]

            results = []
            prev_text = ""
            prev_end = 0.0

            for seg in segments_iter:
                text = seg.text.strip()

                # Skip empty or single-char segments
                if not text or len(text) < 2:
                    continue

                # Edge case: malformed timestamps (end before start) — skip and log
                if seg.end < seg.start:
                    print(f"[Whisper] WARNING: Skipping malformed segment "
                          f"(end {seg.end:.3f} < start {seg.start:.3f}): {text[:50]}")
                    continue

                # Skip Whisper stutter bug (same text within 2 seconds)
                if text == prev_text and (seg.start - prev_end) < 2.0:
                    continue

                # Skip hallucinated text
                text_lower = text.lower()
                if any(pattern in text_lower for pattern in HALLUCINATION_PATTERNS):
                    continue

                # FIX 4: Classify segment as speech, music, or uncertain
                seg_type = classify_segment_type(text, seg)

                # Format music segments with musical note markers
                display_text = text
                if seg_type == "music":
                    display_text = f"\u266a {text} \u266a"  # ♪ text ♪

                # FIX 6: Every segment carries a type field for debugging
                seg_dict = {
                    "start": round(seg.start + start_offset, 3),
                    "end":   round(seg.end   + start_offset, 3),
                    "text":  display_text,
                    "type":  seg_type,
                }

                results.append(seg_dict)
                prev_text = text
                prev_end = seg.end

        finally:
            # Always clean up temp file
            try:
                os.remove(temp_path)
            except Exception:
                pass

        return {
            "segments": results,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": info.duration or 0,
        }


@app.function(gpu="A10", image=whisper_image, timeout=600)
def _cloud_detect_language(audio_bytes: bytes):
    """Runs on cloud GPU. Detects audio language via Whisper's language ID.

    Uses task='transcribe' (Fix 1) to prevent silent translation.

    Args:
        audio_bytes: Raw audio bytes (any format ffmpeg can read).

    Returns:
        dict with 'language' and 'probability' keys.
    """
    from faster_whisper import WhisperModel
    import tempfile as _tempfile

    model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    with _tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        f.flush()
        temp_path = f.name

    try:
        # FIX 1: task="transcribe" forced
        _, info = model.transcribe(temp_path, task="transcribe")
        return {
            "language": info.language,
            "probability": info.language_probability,
        }
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


# ============================================================================
# Helper: Audio Extraction (Fix 3 revised — loudnorm removed for speed)
# ============================================================================

def extract_audio_wav(
    file_path: str,
    start_sec: float = 0.0,
    duration_sec: float = None,
    normalize: bool = False,
) -> bytes:
    """Extract audio from a file as 16kHz mono WAV bytes.

    FIX 3 (revised): loudnorm removed — it took 30-60s per chunk on
    laptop CPU, stalling the pipeline before any GPU call was made.
    The language-aware VAD threshold (0.3 for soft-speech languages)
    compensates for quiet dialogue without normalization.

    Args:
        file_path:    Path to the source audio/video file.
        start_sec:    Start position in seconds.
        duration_sec: Duration to extract (None = full file).
        normalize:    If True, applies lightweight dynaudnorm filter.
                      Default False for speed.

    Returns:
        Raw WAV bytes (16kHz mono PCM).
    """
    cmd = ["ffmpeg"]

    if start_sec > 0:
        cmd.extend(["-ss", str(start_sec)])  # seek BEFORE input = fast seek

    cmd.extend(["-i", file_path])

    if duration_sec is not None:
        cmd.extend(["-t", str(duration_sec)])

    cmd.append("-vn")  # no video

    # Lightweight normalization only when explicitly requested.
    # loudnorm (EBU R128) was removed — too slow on laptop CPU.
    if normalize:
        cmd.extend(["-af", "dynaudnorm=f=150:g=15"])  # fast rolling-window norm

    cmd.extend([
        "-acodec", "pcm_s16le",  # raw PCM — Whisper's native format
        "-ac", "1",              # mono
        "-ar", "16000",          # 16kHz
        "-f", "wav",
        "pipe:1",                # output to stdout
        "-loglevel", "quiet",
    ])

    result = subprocess.run(cmd, capture_output=True)
    return result.stdout


# ============================================================================
# Public API (called by server/main.py and other modules)
# ============================================================================

def transcribe_audio(audio_path, language=None):
    """Transcribe an audio/video file via Modal cloud H100 GPU.

    Extracts audio with EBU R128 normalization (Fix 3), then sends
    to cloud GPU with all v2 fixes applied.

    Args:
        audio_path: path to audio or video file (MP3, WAV, MP4, MKV, etc.)
        language: language code (e.g., 'en'). None = auto-detect.

    Returns:
        (segments, detected_language)
        segments = [{"start": 0.0, "end": 2.5, "text": "Hello", "type": "speech"}, ...]
    """
    print(f"[Whisper] Uploading to cloud H100 GPU (large-v3): {os.path.basename(audio_path)}")

    # FIX 3: Extract audio with loudness normalization
    audio_bytes = extract_audio_wav(audio_path, normalize=True)

    # Edge case 2: guard against empty audio
    if len(audio_bytes) < 1000:
        print(f"[Whisper] WARNING: Audio extraction returned only {len(audio_bytes)} bytes — skipping")
        return [], "unknown"

    size_mb = len(audio_bytes) / (1024 * 1024)
    print(f"[Whisper] Normalized audio: {size_mb:.1f} MB — sending to Modal H100 GPU...")

    # Fire to cloud GPU using the persistent class (model stays loaded)
    transcriber = CloudTranscriber()
    result = transcriber.transcribe_chunk.remote(audio_bytes, 0.0, language)

    segments = result["segments"]
    detected_lang = result["language"]
    prob = result["language_probability"]
    duration = result["duration"]

    duration_str = f"{int(duration // 60):02d}:{int(duration % 60):02d}" if duration else "??:??"
    print(f"[Whisper] Detected language: {detected_lang} ({prob:.0%})")
    print(f"[Whisper] Audio duration: {duration_str}")
    print(f"[Whisper] Transcribed {len(segments)} segments on cloud H100 GPU")

    return segments, detected_lang


def segments_to_srt(segments):
    """Convert transcription segments to SRT format string.

    Input: [{"start": 0.0, "end": 2.5, "text": "Hello", "type": "speech"}, ...]
    Output: SRT formatted string (type field is NOT included in output)
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
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds % 1) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def transcribe_video(video_path, language=None):
    """Full pipeline: send video to cloud H100 GPU → return SRT string.

    Returns:
        dict with keys: srt_content, language, segment_count
    """
    print(f"[Whisper] Processing video: {os.path.basename(video_path)}")

    segments, detected_lang = transcribe_audio(video_path, language=language)
    srt_content = segments_to_srt(segments)

    return {
        "srt_content": srt_content,
        "language": detected_lang,
        "segment_count": len(segments),
    }


# ============================================================================
# Modal Entrypoint (for direct testing: modal run server/transcriber.py)
# ============================================================================

@app.local_entrypoint()
def main():
    """CLI entry point for testing transcription directly.

    Usage:
        modal run server/transcriber.py
    """
    import sys

    # Use a default test file
    test_files = [
        "test_files/test_audio.mp3",
        "test_files/art of sarah.mp4",
    ]

    filepath = None
    for tf in test_files:
        if os.path.exists(tf):
            filepath = tf
            break

    if filepath is None:
        print("❌ No test file found! Put an audio/video file in test_files/")
        return

    print(f"🚀 Testing cloud transcription on: {filepath}")
    print(f"   Sending to Modal H100 GPU with Whisper large-v3 (v2 fixes)...\n")

    # FIX 3: Extract with normalization
    audio_bytes = extract_audio_wav(filepath, normalize=True)

    # Edge case 2: guard against empty audio
    if len(audio_bytes) < 1000:
        print(f"❌ Audio extraction failed — only {len(audio_bytes)} bytes returned")
        return

    size_mb = len(audio_bytes) / (1024 * 1024)
    print(f"   Normalized audio: {size_mb:.1f} MB\n")

    transcriber = CloudTranscriber()
    result = transcriber.transcribe_chunk.remote(audio_bytes, 0.0)

    segments = result["segments"]
    lang = result["language"]
    prob = result["language_probability"]
    duration = result["duration"]

    print(f"\n{'=' * 60}")
    print(f"Language: {lang} ({prob:.0%})")
    print(f"Duration: {int(duration // 60):02d}:{int(duration % 60):02d}")
    print(f"Segments: {len(segments)}")

    # Show type breakdown
    type_counts = {}
    for seg in segments:
        t = seg.get("type", "speech")
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")

    print(f"{'=' * 60}")

    # Build and print the SRT
    srt = segments_to_srt(segments)
    print(srt[:800])
    print("...")
