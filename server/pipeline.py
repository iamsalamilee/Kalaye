"""
Kalaye -- Full Pipeline Orchestrator v3
========================================
The single entry point that runs the entire AI pipeline:
  1. Whisper (cloud GPU) -> full-file transcription with timestamps
  2. AST/YAMNet (cloud) -> environmental sound detection on silence gaps
  3. Merge into a single, unified SRT file (SDH-ready)

Key design decisions:
  - NO CHUNKING: entire audio sent to one GPU container
    (chunking was causing hallucinations at boundaries and context loss)
  - VAD filter enabled (skips silence/music -> faster processing)
  - condition_on_previous_text=True works across the ENTIRE file
    (impossible with chunking — each chunk was isolated)
  - Cold start amortized: model baked into image, loaded once per container

v3 changes:
  - REMOVED chunking entirely (was the #1 source of hallucinations)
  - task="transcribe" forced (no silent translation)
  - Language-aware VAD parameters for soft-speech languages
  - Higher beam_size (10 vs 5) for complex languages
  - Every segment carries a "type" field for pipeline debugging
  - Hallucination filtering, sound batching, gap extraction preserved

Usage:
    modal run server/pipeline.py
    modal run server/pipeline.py --filepath /path/to/movie.mp4
    modal run server/pipeline.py --filepath /path/to/movie.mp4 --language ko
"""

import os
import subprocess
import modal

# ============================================================================
# ONE Modal App -- All cloud functions live here
# ============================================================================

app = modal.App("kalaye-pipeline")

# Heavy GPU image for Whisper large-v3 -- model baked in at build time.
# Must use device='cpu' during build because Modal build-servers have no GPU.
whisper_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04", add_python="3.11"
    )
    .apt_install("ffmpeg")
    .pip_install("faster-whisper")
    .run_commands(
        # Bake model into image. device='cpu' is required here because
        # Modal build servers do not have GPUs attached.
        "python -c \"from faster_whisper import WhisperModel; "
        "WhisperModel('large-v3-turbo', device='cpu')\""
    )
)

# AST/YAMNet sound classifier -- model baked in
sound_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("transformers", "torch", "librosa", "soundfile")
    .env({"HF_HUB_DISABLE_TELEMETRY": "1"})
    .run_commands(
        "python -c \"from transformers import pipeline; "
        "pipeline('audio-classification', model='MIT/ast-finetuned-audioset-10-10-0.4593')\""
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
            "threshold": 0.1,                  # lower energy gate for soft phonemes
            "min_silence_duration_ms": 200,     # aggressive split for granular segments
            "speech_pad_ms": 300,               # moderate padding keeps word boundaries intact
        }
    # Default: must work for ANY language (including auto-detect / multi-language films)
    # Kept low (0.35) so quieter speech in Hindi, Marathi etc. isn't dropped
    return {
        "threshold": 0.25,
        "min_silence_duration_ms": 250,
        "speech_pad_ms": 200,
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


# Music classifier REMOVED — it was marking ~90% of non-English speech
# as music because Whisper has lower word-level confidence on Hindi/Marathi.
# The sound_detector.py already handles environmental sounds separately.


def split_long_segments(segments: list, max_duration: float = 7.0) -> list:
    """Split segments longer than max_duration at word boundaries.

    When task='transcribe' is used (instead of translate), Whisper produces
    fewer, longer segments because it doesn't re-segment into natural English
    sentence boundaries. This post-processor splits oversized segments at
    word timestamps to recover the finer granularity that translate mode gave.

    If a segment has no word-level timestamps, it is kept as-is.

    Words are expected as dicts: {"word": str, "start": float, "end": float}
    (serialized from faster-whisper Word objects in transcribe_chunk).

    Args:
        segments: List of segment dicts with optional 'words' key.
        max_duration: Maximum duration in seconds before splitting.

    Returns:
        New list of segment dicts with long segments split.
    """
    output = []
    for seg in segments:
        duration = seg["end"] - seg["start"]
        words = seg.get("words")

        # Short enough or no word data — keep as is
        if duration <= max_duration or not words:
            output.append(seg)
            continue

        # Split into sub-segments of roughly max_duration each
        current_words = []
        sub_start = words[0].get("start", seg["start"])

        for w in words:
            w_start = w.get("start")
            w_end   = w.get("end")
            w_text  = w.get("word", "")

            if w_start is None or w_end is None:
                current_words.append(w_text)
                continue

            current_words.append(w_text)
            elapsed = w_end - sub_start

            if elapsed >= max_duration and len(current_words) >= 2:
                # Flush current sub-segment
                text = "".join(current_words).strip()
                output.append({
                    "start": round(sub_start, 3),
                    "end":   round(w_end, 3),
                    "text":  text,
                    "type":  "speech",
                })
                current_words = []
                sub_start = w_end

        # Flush remaining words
        if current_words:
            last_end = seg["end"]
            for w in reversed(words):
                if w.get("end") is not None:
                    last_end = w["end"]
                    break
            text = "".join(current_words).strip()
            if text:
                output.append({
                    "start": round(sub_start, 3),
                    "end":   round(last_end, 3),
                    "text":  text,
                    "type":  "speech",
                })

    return output


# ============================================================================
# Cloud Function 1: Whisper Transcription (Parallel Chunk Mode)
# ============================================================================

@app.cls(
    gpu="A10",                        # $1.10/hr — faster than T4, plenty of VRAM for large-v3-turbo
    image=whisper_image,
    timeout=1800,                     # 30 min max — full movie in one shot
    scaledown_window=30,              # 30s warmth — don't pay for idle containers
)
class WhisperTranscriber:

    @modal.enter()
    def load_model(self):
        """Runs ONCE when container starts. All requests reuse this model."""
        from faster_whisper import WhisperModel
        self.model = WhisperModel(
            "large-v3-turbo",            # distilled large-v3: 8x faster, same languages
            device="cuda",
            compute_type="float16",         # fastest on T4, minimal quality loss
        )
        print("[Whisper] large-v3-turbo loaded on GPU -- container is warm")

    @modal.method()
    def transcribe_chunk(
        self,
        audio_bytes: bytes,
        start_offset: float,
        language: str = None,
    ):
        """Transcribe one audio chunk and shift all timestamps by start_offset.

        Applies Fixes 1-6:
          - task='transcribe' forced (Fix 1)
          - Language-aware VAD params (Fix 2)
          - beam_size per language (Fix 5)
          - Music detection + formatting (Fix 4)
          - Type field on every segment (Fix 6)

        Args:
            audio_bytes:  Raw WAV bytes for this chunk (16kHz mono PCM).
            start_offset: Where this chunk starts in the full movie (seconds).
            language:     ISO 639-1 code to force language, or None for auto.

        Returns:
            dict with segments (timestamp-shifted), detected language, and
            language_probability.
        """
        import tempfile
        import os

        # Write audio bytes to temp file for Whisper (Opus or WAV both work)
        suffix = ".ogg" if audio_bytes[:4] == b'OggS' else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            # FIX 2: Get language-tuned VAD parameters
            vad_params = get_vad_params_for_language(language)

            # FIX 5: Get language-tuned beam size
            beam_size = get_beam_size_for_language(language)

            segments_iter, info = self.model.transcribe(
                temp_path,
                task="transcribe",                     # FIX 1: NEVER translate, always transcribe
                language=language,
                beam_size=beam_size,                    # FIX 5: 10 for complex languages, 5 otherwise
                word_timestamps=True,
                condition_on_previous_text=True,        # keeps language consistent across segments
                vad_filter=True,                        # skip silence and music
                vad_parameters=vad_params,              # FIX 2: language-aware thresholds
                hallucination_silence_threshold=2.0,    # lower with condition_on_previous=True to prevent loops
            )

            # Known Whisper hallucination strings during silence/music/credits.
            # FIX 7: Extended with music-specific hallucination patterns.
            HALLUCINATION_PATTERNS = [
                "subtitle", "\uc790\ub9c9", "\ud55c\uae00\uc790\ub9c9", "\uc124\uc815\uc5d0\uc11c",
                "\uac10\uc0ac\ud569\ub2c8\ub2e4", "thank you", "thanks for watching",
                "subscribe", "\uad6c\ub3c5", "\uc88b\uc544\uc694",
                "by \ud55c", "by han", "\uc544\uba58", "amen",
                "www.", ".com", "http",
                "\u266b", "\ud83c\udfb5", "la la", "na na", "hmm hmm",
                "foreign",                              # Whisper outputs this for untranscribable audio
                "\u00a9 transcript", "emily beynon",     # OCR-style hallucinations from credits
            ]

            results = []
            prev_text = ""
            prev_end = 0.0

            for seg in segments_iter:
                text = seg.text.strip()

                # Skip empty or single-char segments
                if not text or len(text) < 2:
                    continue

                # Edge case: malformed timestamps (end before start) -- skip and log
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

                # All segments are treated as speech — music classifier removed
                # because it was marking ~90% of Hindi/Marathi as music.
                seg_dict = {
                    "start": round(seg.start + start_offset, 3),
                    "end":   round(seg.end   + start_offset, 3),
                    "text":  text,
                    "type":  "speech",
                }
                # Attach word timestamps so split_long_segments can break
                # oversized segments at natural word boundaries.
                # Serialize Word objects to plain dicts with start_offset applied
                # so they survive Modal's RPC serialization.
                if hasattr(seg, "words") and seg.words:
                    seg_dict["words"] = [
                        {
                            "word":  w.word,
                            "start": round(w.start + start_offset, 3),
                            "end":   round(w.end   + start_offset, 3),
                        }
                        for w in seg.words
                        if hasattr(w, "word") and hasattr(w, "start") and hasattr(w, "end")
                    ]

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

    @modal.method()
    def transcribe(self, audio_bytes: bytes, language: str = None):
        """Legacy single-file transcription (kept for backward compatibility).

        For new code, use transcribe_chunk with parallelization instead.
        """
        return self.transcribe_chunk(audio_bytes, 0.0, language)


# ============================================================================
# Cloud Function 2: Environmental Sound Classification
# ============================================================================

# Labels to ignore -- too generic or overlap with the speech pipeline
IGNORED_LABELS = {
    "speech", "narration, monologue", "conversation",
    "silence", "white noise", "pink noise",
    "inside, small room", "inside, large room or hall",
    "background music", "soundtrack",
    "crowd", "chatter", "hubbub, speech noise, speech babble",
    "noise", "environmental noise", "outside, urban or manmade",
    "outside, rural or natural",
}

# Music-related labels that get ♪ ♪ formatting instead of [Label]
MUSIC_LABELS = {"music", "singing", "song", "musical instrument"}

MIN_CONFIDENCE = 0.15  # Only tag sounds we are fairly sure about


@app.cls(
    gpu="T4",                         # $0.59/hr — AST model is tiny
    image=sound_image,
    timeout=300,
    scaledown_window=30,              # minimal idle billing
)
class SoundClassifier:

    @modal.enter()
    def load_model(self):
        """Runs ONCE when container starts."""
        from transformers import pipeline
        import warnings
        warnings.filterwarnings("ignore")

        self.classifier = pipeline(
            "audio-classification",
            model="MIT/ast-finetuned-audioset-10-10-0.4593",
            device=0,  # GPU
        )
        print("[SoundClassifier] Model loaded on GPU -- container is warm")

    @modal.method()
    def classify(self, audio_bytes: bytes, top_k: int = 5):
        """Classifies environmental sounds in an audio chunk.

        Args:
            audio_bytes: Raw WAV bytes.
            top_k:       Number of top predictions to return.

        Returns:
            list of dicts with 'label' and 'score' keys.
        """
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            predictions = self.classifier(temp_path, top_k=top_k)
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass

        return predictions


# ============================================================================
# Helper Functions
# ============================================================================

def extract_audio_chunk_opus(
    file_path: str,
    start_sec: float,
    duration_sec: float,
) -> bytes:
    """Extract a chunk of audio as compressed OGG Opus bytes for Whisper.

    Opus at 16kbps mono is ~16x smaller than WAV PCM while losing
    nothing Whisper needs. Faster-whisper decodes Opus natively via
    ffmpeg on the GPU side, so no quality loss for transcription.

    Size comparison for a 10-min chunk:
      WAV PCM:  ~19.2 MB  (16kHz × 16-bit × mono × 600s)
      Opus 16k: ~1.2 MB   (16kbps × 600s)

    Args:
        file_path:    Path to the source audio/video file.
        start_sec:    Start position in seconds.
        duration_sec: Duration to extract in seconds.

    Returns:
        OGG Opus bytes (16kHz mono, 16kbps).
    """
    cmd = [
        "ffmpeg",
        "-ss", str(start_sec),         # seek BEFORE input = fast seek
        "-i", file_path,
        "-t",  str(duration_sec),
        "-vn",                          # no video
        "-acodec", "libopus",           # Opus codec — tiny size, Whisper reads it
        "-ac", "1",                     # mono
        "-ar", "16000",                 # 16kHz — Whisper's native sample rate
        "-b:a", "16k",                  # 16kbps — more than enough for speech
        "-f", "ogg",                    # OGG container
        "pipe:1",                       # output to stdout
        "-loglevel", "quiet",
    ]

    result = subprocess.run(cmd, capture_output=True)
    return result.stdout


def extract_audio_chunk_wav(
    file_path: str,
    start_sec: float,
    duration_sec: float,
) -> bytes:
    """Extract a chunk of audio as WAV PCM bytes for the sound classifier.

    The AST sound classifier needs uncompressed audio with original
    energy levels to correctly identify environmental sounds.
    Only used for silence gap analysis (small chunks, few seconds each).

    Args:
        file_path:    Path to the source audio/video file.
        start_sec:    Start position in seconds.
        duration_sec: Duration to extract in seconds.

    Returns:
        Raw WAV bytes (16kHz mono PCM).
    """
    cmd = [
        "ffmpeg",
        "-ss", str(start_sec),         # seek BEFORE input = fast seek
        "-i", file_path,
        "-t",  str(duration_sec),
        "-vn",                          # no video
        "-acodec", "pcm_s16le",         # raw PCM
        "-ac", "1",                     # mono
        "-ar", "16000",                 # 16kHz
        "-f", "wav",
        "pipe:1",                       # output to stdout
        "-loglevel", "quiet",
    ]

    result = subprocess.run(cmd, capture_output=True)
    return result.stdout


def segments_to_srt(segments: list) -> str:
    """Convert a list of segment dicts to a valid SRT string.

    Args:
        segments: List of dicts with 'start', 'end', 'text' keys.

    Returns:
        Complete SRT file content as a string.
    """
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_srt_timestamp(seg["start"])
        end   = _format_srt_timestamp(seg["end"])
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def _format_srt_timestamp(seconds: float) -> str:
    """Convert float seconds to SRT format: HH:MM:SS,mmm"""
    seconds = max(0.0, seconds)
    hours   = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs    = int(seconds % 60)
    millis  = int(round((seconds % 1) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _get_duration(file_path: str) -> float:
    """Get audio/video duration in seconds using ffprobe.

    Args:
        file_path: Path to the media file.

    Returns:
        Duration in seconds, or 300.0 as fallback.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        file_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return 300.0  # fallback: assume 5 minutes


def find_silence_gaps(segments: list, total_duration: float, min_gap_sec: float = 1.5) -> list:
    """Find time gaps between speech segments larger than min_gap_sec.

    Args:
        segments:       Sorted list of segment dicts with 'start' and 'end'.
        total_duration: Total duration of the media file in seconds.
        min_gap_sec:    Minimum gap duration to report.

    Returns:
        List of (gap_start, gap_end) tuples.
    """
    gaps = []

    if not segments:
        return [(0.0, total_duration)] if total_duration > min_gap_sec else []

    # Gap before first segment
    if segments[0]["start"] > min_gap_sec:
        gaps.append((0.0, segments[0]["start"]))

    # Gaps between segments
    for i in range(len(segments) - 1):
        gap_start = segments[i]["end"]
        gap_end   = segments[i + 1]["start"]
        if (gap_end - gap_start) >= min_gap_sec:
            gaps.append((gap_start, gap_end))

    # Gap after last segment
    if total_duration and (total_duration - segments[-1]["end"]) > min_gap_sec:
        gaps.append((segments[-1]["end"], total_duration))

    return gaps


# remove_overlap_duplicates REMOVED in v3 — no chunking = no overlap duplicates


# ============================================================================
# The Full Pipeline
# ============================================================================

def run_pipeline(file_path: str, language: str = None) -> dict:
    """Run the complete Kalaye pipeline on a video/audio file.

    v3: No chunking — entire audio sent to ONE GPU container.
    This eliminates hallucinations caused by chunk boundaries,
    context loss between chunks, and overlap deduplication artifacts.

    Steps:
      1. Get duration via ffprobe
      2. Extract full audio as compressed Opus
      3. Send entire file to ONE Whisper GPU container
      4. Split long segments at word boundaries
      5. Find silence gaps -> sound classifier
      6. Merge dialogue + sound tags -> final SDH SRT

    Args:
        file_path: Path to any video or audio file ffmpeg can read.
        language:  ISO 639-1 code to force (e.g. 'ko', 'en'), or None for auto.

    Returns:
        dict with keys:
            srt_content    -- complete SRT file as string
            language       -- detected primary language code
            segment_count  -- total entries in SRT (dialogue + sounds)
            dialogue_count -- dialogue-only segment count
            sound_tags     -- list of unique environmental sound labels found
    """
    print(f"\n{'=' * 62}")
    print(f"  KALAYE PIPELINE v3 (no chunking)  --  {os.path.basename(file_path)}")
    print(f"{'=' * 62}\n")

    # ---- Get duration directly from the original file ----
    duration = _get_duration(file_path)
    duration_str = f"{int(duration // 60):02d}:{int(duration % 60):02d}"
    print(f"[Pre-step] Duration: {duration_str} ({duration:.0f} seconds)")
    if language:
        print(f"           Language forced: {language}")
        print(f"           VAD mode: {'soft-speech (low threshold)' if language in SOFT_SPEECH_LANGUAGES else 'standard'}")
        print(f"           Beam size: {get_beam_size_for_language(language)}")
    else:
        print(f"           Language: auto-detect")
    print()

    # ---- Step 1: Full-file Whisper transcription (single GPU) ----
    print("[Step 1/3] [GPU] Extracting full audio (Opus 16kbps)...")

    opus_bytes = extract_audio_chunk_opus(file_path, 0.0, duration)

    if len(opus_bytes) < 500:
        print(f"           [!] Audio extraction failed ({len(opus_bytes)} bytes)")
        return {
            "srt_content":    "",
            "language":       "unknown",
            "segment_count":  0,
            "dialogue_count": 0,
            "sound_tags":     [],
        }

    total_mb = len(opus_bytes) / (1024 * 1024)
    print(f"           Audio size: {total_mb:.1f} MB")
    print(f"           ⏳ Sending entire file to ONE GPU (no chunking = no boundary hallucinations)...\n")

    whisper = WhisperTranscriber()

    try:
        result = whisper.transcribe_chunk.remote(opus_bytes, 0.0, language)
    except Exception as e:
        print(f"           [!] Transcription FAILED: {e}")
        return {
            "srt_content":    "",
            "language":       "unknown",
            "segment_count":  0,
            "dialogue_count": 0,
            "sound_tags":     [],
        }

    all_dialogue  = result["segments"]
    detected_lang = result.get("language", "unknown")
    lang_prob     = result.get("language_probability", 0.0)

    # Sort by time (should already be sorted, but ensure it)
    all_dialogue.sort(key=lambda x: x["start"])

    # Split long segments at word boundaries.
    # task='transcribe' produces fewer, longer segments than translate mode.
    # This recovers the fine-grained subtitle entries users expect.
    pre_split_count = len(all_dialogue)
    all_dialogue = split_long_segments(all_dialogue, max_duration=7.0)

    # Strip word timestamps from final output (not needed in SRT)
    for seg in all_dialogue:
        seg.pop("words", None)

    print(f"           [OK] Transcription complete")
    print(f"           [OK] {pre_split_count} segments from Whisper")
    if len(all_dialogue) > pre_split_count:
        print(f"           [OK] {len(all_dialogue)} segments after splitting long entries")
    print(f"           Primary language: {detected_lang} ({lang_prob:.0%})")


    print()

    # ---- Step 2: Environmental sound detection on silence gaps ----
    print("[Step 2/3] [GPU] Analyzing silence gaps for environmental sounds...")

    gaps = find_silence_gaps(all_dialogue, duration, min_gap_sec=3.0)

    # Cap at 40 gaps -- sort by length descending, take largest, re-sort by time
    if len(gaps) > 40:
        gaps.sort(key=lambda g: g[1] - g[0], reverse=True)
        gaps = gaps[:40]
        gaps.sort(key=lambda g: g[0])

    print(f"           Found {len(gaps)} silence gaps >= 3 seconds")

    sound_events = []

    if gaps:
        # Extract WAV for each gap locally -- FIX 3: normalize=False for sound classifier
        gap_chunks = []
        gap_meta   = []

        for gap_start, gap_end in gaps:
            chunk_duration = min(gap_end - gap_start, 10.0)
            wav_bytes = extract_audio_chunk_wav(
                file_path,              # use original file
                gap_start,
                chunk_duration,
            )
            # Edge case 2: Skip empty chunks
            if len(wav_bytes) > 1000:
                gap_chunks.append(wav_bytes)
                gap_meta.append((gap_start, gap_end))

        if gap_chunks:
            print(f"           Sending {len(gap_chunks)} gaps to sound classifier (batches of 10)...")

            try:
                all_predictions = []
                batch_size      = 10
                classifier      = SoundClassifier()

                for i in range(0, len(gap_chunks), batch_size):
                    batch       = gap_chunks[i : i + batch_size]
                    batch_preds = list(
                        classifier.classify.map(batch, return_exceptions=True)
                    )
                    all_predictions.extend(batch_preds)

                for idx, predictions in enumerate(all_predictions):
                    if isinstance(predictions, Exception):
                        print(f"           [!] Sound gap {idx} failed: {predictions}")
                        continue

                    gap_start, gap_end = gap_meta[idx]

                    for pred in predictions:
                        label = pred["label"].lower()
                        if label in IGNORED_LABELS:
                            continue
                        if pred["score"] < MIN_CONFIDENCE:
                            continue

                        # FIX 4: Music labels get ♪ ♪ format instead of [Label]
                        is_music = any(m in label for m in MUSIC_LABELS)
                        if is_music:
                            tag_text = "\u266a \u266a"            # ♪ ♪ for instrumental music
                            tag_type = "music"
                        else:
                            # Use first part of label (e.g. "Gunshot, gunfire" -> "Gunshot")
                            tag_label = pred["label"].split(",")[0].strip().title()
                            tag_text  = f"[{tag_label}]"
                            tag_type  = "sound"

                        # FIX 6: Sound events carry a type field
                        sound_events.append({
                            "start": gap_start,
                            "end":   min(gap_start + 2.0, gap_end),
                            "text":  tag_text,
                            "type":  tag_type,
                        })

            except Exception as e:
                print(f"           [!] Sound detection failed entirely: {e}")

    # Deduplicate sound events (same tag at same time)
    unique_sounds = []
    seen_keys     = set()
    for event in sound_events:
        key = f"{event['text']}_{event['start']:.0f}"
        if key not in seen_keys:
            unique_sounds.append(event)
            seen_keys.add(key)

    print(f"           [OK] {len(unique_sounds)} environmental sound tags detected\n")

    # ---- Step 3: Merge dialogue + sounds -> final SRT ----
    print("[Step 3/3] [LOCAL] Merging dialogue + sounds into unified SRT...")

    # Edge case 7: Keep both dialogue and music even if they overlap in time
    all_entries = all_dialogue + unique_sounds
    all_entries.sort(key=lambda x: x["start"])

    srt_content   = segments_to_srt(all_entries)
    total_entries = len(all_entries)

    print(f"           [OK] Final SRT: {total_entries} total entries")
    print(f"               |-- {len(all_dialogue)} dialogue segments")
    print(f"               |-- {len(unique_sounds)} sound tags")
    print(f"\n{'=' * 62}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'=' * 62}\n")

    tag_list = sorted(set(e["text"] for e in unique_sounds))

    return {
        "srt_content":    srt_content,
        "language":       detected_lang,
        "segment_count":  total_entries,
        "dialogue_count": len(all_dialogue),
        "sound_tags":     tag_list,
    }


# ============================================================================
# Modal Local Entrypoint (for testing: modal run server/pipeline.py)
# ============================================================================

@app.local_entrypoint()
def main(
    filepath:  str = None,
    language:  str = None,
    json_out:  str = None,
):
    """CLI entrypoint for testing the pipeline locally via Modal.

    Examples:
        modal run server/pipeline.py
        modal run server/pipeline.py --filepath movie.mp4
        modal run server/pipeline.py --filepath movie.mp4 --language ko
        modal run server/pipeline.py --filepath movie.mp4 --json-out result.json
    """
    # Fallback to test files if no filepath provided
    if not filepath:
        test_files = [
            "test_files/art of sarah.mp4",
            "test_files/test_audio.mp3",
        ]
        for tf in test_files:
            if os.path.exists(tf):
                filepath = tf
                break

    if not filepath or not os.path.exists(filepath):
        print(f"  Error: File not found -- {filepath}")
        print("  Usage: modal run server/pipeline.py --filepath /path/to/movie.mp4")
        return

    result = run_pipeline(filepath, language=language)

    # JSON output mode (used by GUI to receive results)
    if json_out:
        import json
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"JSON result written to: {json_out}")
        return

    # Standard CLI mode -- save SRT and print preview
    output_path = "test_files/output.srt"
    os.makedirs("test_files", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result["srt_content"])

    print(f"SRT saved to: {output_path}")
    print(f"\nSRT Preview (first 2000 chars):")
    print("-" * 50)
    print(result["srt_content"][:2000])
    print("-" * 50)
    print(f"\nLanguage detected : {result['language']}")
    print(f"Dialogue segments : {result['dialogue_count']}")
    print(f"Sound tags        : {', '.join(result['sound_tags']) if result['sound_tags'] else '(none)'}")
    print(f"Total SRT entries : {result['segment_count']}")