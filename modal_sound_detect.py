"""
Kalaye — Standalone Sound Detection Test (Modal H100 GPU) v2
==============================================================
Quick standalone test for environmental sound classification on cloud GPU.

v2 fixes applied:
  - H100 GPU for faster inference
  - Model baked into image (no cold-start download)
  - Music-related labels get ♪ ♪ formatting (Fix 4)

Usage:
    modal run modal_sound_detect.py
"""

import modal

app = modal.App("kalaye-sound-detect-test")

# Setup cloud environment for sound classification with baked-in model
sound_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    # Transformers + torch handle the AST/YAMNet AI models
    .pip_install("transformers", "torch", "librosa", "soundfile")
    .env({"HF_HUB_DISABLE_TELEMETRY": "1"})
    .run_commands(
        # Bake model into image — no download at runtime.
        "python -c \"from transformers import pipeline; "
        "pipeline('audio-classification', model='MIT/ast-finetuned-audioset-10-10-0.4593')\""
    )
)

# Music-related labels that get ♪ ♪ formatting instead of [Label]
MUSIC_LABELS = {"music", "singing", "song", "musical instrument"}


@app.function(gpu="H100", image=sound_image, timeout=300)
def classify_audio(audio_bytes: bytes):
    """Classify environmental sounds in audio on cloud H100 GPU.

    Uses the AST AudioSet model — the modern, highly accurate upgrade
    to Google's YAMNet. Knows 500+ environmental sounds (gunshots,
    doors, sirens, music, rain, etc.)

    Args:
        audio_bytes: Raw audio bytes (any format).

    Returns:
        list of dicts with 'label' and 'score' keys.
    """
    from transformers import pipeline
    import tempfile
    import warnings
    warnings.filterwarnings("ignore")  # hide messy AI warnings

    # Load the AST AudioSet model
    classifier = pipeline(
        "audio-classification",
        model="MIT/ast-finetuned-audioset-10-10-0.4593",
        device=0,  # GPU
    )

    # Save the bytes to a temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        f.flush()

        print("🧠 Cloud H100 AI is listening to the environment...")
        # The pipeline listens to the audio and returns top 5 sounds
        predictions = classifier(f.name)

        return predictions


@app.local_entrypoint()
def main():
    """CLI entry point for standalone sound detection test.

    Usage:
        modal run modal_sound_detect.py
    """
    import os

    audio_file = "test_files/test_audio.mp3"

    if not os.path.exists(audio_file):
        print("❌ Error: Please put a short audio file in the test_files "
              "folder and name it 'test_audio.mp3'")
        return

    audio_bytes = open(audio_file, "rb").read()

    # Edge case 2: guard against empty audio
    if len(audio_bytes) < 1000:
        print(f"❌ Audio file too small ({len(audio_bytes)} bytes)")
        return

    size_kb = len(audio_bytes) / 1024
    print(f"☁️  Uploading audio to H100 GPU for sound classification ({size_kb:.0f} KB)...")

    # Send it to the cloud H100 GPU
    results = classify_audio.remote(audio_bytes)

    print("\n🎧 What the AI heard (Top 5 Matches):")
    print("-" * 40)
    for sound in results:
        confidence = sound['score'] * 100
        label_lower = sound['label'].lower()

        # FIX 4: Music-related labels get ♪ ♪ formatting
        is_music = any(m in label_lower for m in MUSIC_LABELS)
        if is_music:
            print(f"  ♪ ♪  ({confidence:.1f}%)  — {sound['label'].title()}")
        else:
            # Print it how it would look in SDH subtitles
            print(f"  [{sound['label'].title()}]  (Confidence: {confidence:.1f}%)")
    print("-" * 40)
