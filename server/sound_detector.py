"""
Kalaye — Cloud Sound Detector (Modal GPU) v2
================================================
Classifies environmental sounds in audio using the AST
(Audio Spectrogram Transformer) model fine-tuned on AudioSet.

Detects 500+ sound types (gunshots, doors, music, sirens, rain, etc.)
for generating SDH (Subtitles for the Deaf/Hard-of-Hearing) tags.

v2 fixes applied:
  - T4 GPU (AST is lightweight — T4 is sufficient)
  - Model baked into image (no cold-start download)
  - Music-related labels get ♪ ♪ formatting instead of [Label]
  - Every sound event has a "type" field for debugging (Fix 6)
  - Lower MIN_CONFIDENCE (0.15) matches pipeline.py
  - Expanded IGNORED_LABELS to avoid tagging generic sounds

All processing runs on Modal's serverless cloud — no local GPU needed.
"""

import os
import modal

# ============================================================================
# Modal App & Cloud Environment
# ============================================================================

app = modal.App("kalaye-sound-detect")

# AST/YAMNet sound classifier — model baked in at build time
sound_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("transformers", "torch", "librosa", "soundfile")
    .env({"HF_HUB_DISABLE_TELEMETRY": "1"})
    .run_commands(
        # Bake model into image so containers start instantly (no download).
        "python -c \"from transformers import pipeline; "
        "pipeline('audio-classification', model='MIT/ast-finetuned-audioset-10-10-0.4593')\""
    )
)

# Minimum confidence threshold — sounds below this are ignored.
# Lowered from 0.50 to 0.15 in v2 to catch more subtle environmental sounds.
MIN_CONFIDENCE = 0.15

# Sounds that are just "noise" and not useful in subtitles.
# Expanded in v2 to match pipeline.py's comprehensive ignore list.
IGNORED_LABELS = {
    "speech", "narration, monologue", "conversation",
    "silence", "white noise", "pink noise", "static",
    "inside, small room", "inside, large room or hall",
    "music", "background music", "soundtrack",
    "crowd", "chatter", "hubbub, speech noise, speech babble",
    "noise", "environmental noise", "outside, urban or manmade",
    "outside, rural or natural", "vehicle",
}

# Music-related labels that get ♪ ♪ formatting instead of [Label].
# FIX 4: These indicate instrumental music (no lyrics to transcribe).
MUSIC_LABELS = {"music", "singing", "song", "musical instrument"}


@app.cls(
    gpu="T4",
    image=sound_image,
    timeout=300,
    scaledown_window=120,         # keep warm for 2 min between requests
)
class CloudSoundClassifier:
    """AST sound classifier on T4 GPU with baked-in model."""

    @modal.enter()
    def load_model(self):
        """Runs ONCE when container starts. All requests reuse this model."""
        from transformers import pipeline
        import warnings
        warnings.filterwarnings("ignore")

        self.classifier = pipeline(
            "audio-classification",
            model="MIT/ast-finetuned-audioset-10-10-0.4593",
            device=0,  # GPU
        )
        print("[SoundClassifier] Model loaded on T4 GPU — container is warm")

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

        # Edge case 2: skip empty/corrupt audio chunks
        if len(audio_bytes) < 1000:
            print(f"[SoundClassifier] WARNING: Skipping tiny chunk ({len(audio_bytes)} bytes)")
            return []

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


# Legacy function-based endpoint (kept for backward compat with old callers)
@app.function(gpu="T4", image=sound_image, timeout=300)
def _cloud_classify(audio_bytes: bytes, top_k: int = 5):
    """Runs on Modal cloud. Classifies environmental sounds using AST.

    Returns the top-k predictions with labels and confidence scores.
    """
    from transformers import pipeline
    import tempfile
    import warnings
    warnings.filterwarnings("ignore")

    classifier = pipeline(
        "audio-classification",
        model="MIT/ast-finetuned-audioset-10-10-0.4593",
        device=0,  # GPU
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        f.flush()
        temp_path = f.name

    try:
        predictions = classifier(temp_path, top_k=top_k)
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

    return predictions


# ============================================================================
# Public API (called by server/main.py and other modules)
# ============================================================================

def classify_audio(audio_path, top_k=5):
    """Classify environmental sounds in an audio file via Modal cloud H100.

    Args:
        audio_path: path to audio file (WAV, MP3, etc.)
        top_k: number of top predictions to return

    Returns:
        List of dicts: [{"label": "Gunshot", "score": 0.87, "type": "sound"}, ...]
    """
    print(f"[SoundDetect] Uploading to cloud T4: {os.path.basename(audio_path)}")

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    # Edge case 2: guard against empty audio
    if len(audio_bytes) < 1000:
        print(f"[SoundDetect] WARNING: Audio file too small ({len(audio_bytes)} bytes) — skipping")
        return []

    size_kb = len(audio_bytes) / 1024
    print(f"[SoundDetect] File size: {size_kb:.0f} KB — classifying on T4 GPU...")

    classifier = CloudSoundClassifier()
    predictions = classifier.classify.remote(audio_bytes, top_k=top_k)

    # Filter out speech/noise and low-confidence results
    filtered = []
    for pred in predictions:
        label = pred["label"].lower()
        if label in IGNORED_LABELS:
            continue
        if pred["score"] < MIN_CONFIDENCE:
            continue

        # FIX 4: Music labels get ♪ ♪ formatting instead of [Label]
        is_music = any(m in label for m in MUSIC_LABELS)
        if is_music:
            filtered.append({
                "label": "\u266a \u266a",          # ♪ ♪ for instrumental music
                "score": pred["score"],
                "type": "music",                    # FIX 6: type field
            })
        else:
            filtered.append({
                "label": pred["label"].split(",")[0].strip().title(),
                "score": pred["score"],
                "type": "sound",                    # FIX 6: type field
            })

    if filtered:
        print(f"[SoundDetect] Detected: {', '.join(f['label'] for f in filtered)}")
    else:
        print(f"[SoundDetect] No significant environmental sounds detected.")

    return filtered


def sounds_to_sdh_tags(sounds):
    """Convert classified sounds into SDH subtitle tags.

    Input: [{"label": "Gunshot", "score": 0.87, "type": "sound"}]
    Output: ["[Gunshot]", ...] or ["♪ ♪", ...] for music

    Uses only the first word of multi-word labels for cleaner subtitles.
    Music labels are already formatted as ♪ ♪ by classify_audio().
    """
    tags = []
    for sound in sounds:
        label = sound["label"]
        # Music labels already have ♪ ♪ format from classify_audio()
        if sound.get("type") == "music":
            tag = label  # already "♪ ♪"
        else:
            tag = f"[{label}]"
        if tag not in tags:
            tags.append(tag)
    return tags


def classify_video_audio(video_path, chunk_duration_sec=5):
    """Extract audio from a video file and classify environmental sounds.

    For a full movie, this would be called on specific chunks
    (e.g., segments where Whisper detected no speech).

    Args:
        video_path: path to video file
        chunk_duration_sec: how many seconds of audio to analyze

    Returns:
        List of SDH tags: ["[Gunshot]", "[Door Slam]", "♪ ♪", ...]
    """
    import subprocess
    import tempfile

    # Extract a short audio chunk using ffmpeg (NO normalization — Fix 3:
    # the classifier needs original energy levels to correctly identify sounds)
    tmp_audio = os.path.join(
        tempfile.gettempdir(),
        f"kalaye_sound_{os.getpid()}.wav"
    )

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", "16000",
        "-t", str(chunk_duration_sec),
        tmp_audio,
        "-y",
        "-loglevel", "quiet",
    ]

    subprocess.run(cmd, check=True)

    try:
        sounds = classify_audio(tmp_audio)
        return sounds_to_sdh_tags(sounds)
    finally:
        if os.path.exists(tmp_audio):
            os.remove(tmp_audio)


# ============================================================================
# Modal Entrypoint (for direct testing: modal run server/sound_detector.py)
# ============================================================================

@app.local_entrypoint()
def main():
    """CLI entry point for testing sound detection directly.

    Usage:
        modal run server/sound_detector.py
    """
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

    print(f"🎧 Testing cloud sound detection on: {filepath}")

    with open(filepath, "rb") as f:
        audio_bytes = f.read()

    # Edge case 2: guard
    if len(audio_bytes) < 1000:
        print(f"❌ Audio file too small ({len(audio_bytes)} bytes)")
        return

    size_kb = len(audio_bytes) / 1024
    print(f"   File size: {size_kb:.0f} KB — sending to Modal T4 GPU...\n")

    classifier = CloudSoundClassifier()
    predictions = classifier.classify.remote(audio_bytes)

    print(f"{'─' * 40}")
    print(f"🔊 Raw AI Detections (Top 5):")
    for pred in predictions:
        confidence = pred["score"] * 100
        print(f"  [{pred['label'].title()}]  ({confidence:.1f}%)")
    print(f"{'─' * 40}")

    # Show filtered SDH tags with music formatting
    filtered = []
    for pred in predictions:
        label = pred["label"].lower()
        if label in IGNORED_LABELS:
            continue
        if pred["score"] < MIN_CONFIDENCE:
            continue

        is_music = any(m in label for m in MUSIC_LABELS)
        if is_music:
            filtered.append({"label": "\u266a \u266a", "score": pred["score"], "type": "music"})
        else:
            filtered.append({
                "label": pred["label"].split(",")[0].strip().title(),
                "score": pred["score"],
                "type": "sound",
            })

    tags = sounds_to_sdh_tags(filtered)
    print(f"\n📺 SDH Subtitle Tags: {' '.join(tags) if tags else '(no environmental sounds)'}")
