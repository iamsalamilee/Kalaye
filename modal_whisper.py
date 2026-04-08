"""
Kalaye — Standalone Whisper Test (Modal H100 GPU) v2
=====================================================
Quick standalone test for Whisper transcription on cloud H100 GPU.

v2 fixes applied:
  - FIX 1: task="transcribe" forced — prevents silent translation
  - H100 GPU for faster inference
  - Model baked into image (no cold-start download)
  - Word timestamps enabled for music detection

Usage:
    modal run modal_whisper.py
"""

import modal

app = modal.App("kalaye-whisper-test")

# Set up the cloud environment with model baked in at build time.
# Using Nvidia's official CUDA image so libcublas is included.
whisper_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04", add_python="3.11"
    )
    .apt_install("ffmpeg")
    .pip_install("faster-whisper")
    .run_commands(
        # Bake model into image — no download at runtime.
        # device='cpu' required during build (no GPU on build servers).
        "python -c \"from faster_whisper import WhisperModel; "
        "WhisperModel('large-v3', device='cpu')\""
    )
)


@app.function(gpu="H100", image=whisper_image, timeout=600)
def transcribe(audio_bytes: bytes):
    """Transcribe audio on cloud H100 GPU using Faster-Whisper large-v3.

    FIX 1: task='transcribe' is forced so Whisper NEVER silently translates.
    All translation is handled downstream by Gemini.

    Args:
        audio_bytes: Raw audio bytes (any format ffmpeg can read).

    Returns:
        str: Concatenated transcription text.
    """
    from faster_whisper import WhisperModel
    import tempfile

    # Using large-v3 for maximum accuracy
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        f.write(audio_bytes)
        f.flush()

        segments, info = model.transcribe(
            f.name,
            task="transcribe",                  # FIX 1: NEVER translate
            vad_filter=True,                    # filter silence for cleaner output
            word_timestamps=True,               # needed for music detection
            condition_on_previous_text=False,    # prevents hallucination carry-over
        )
        return " ".join([seg.text for seg in segments])


@app.local_entrypoint()
def main():
    """CLI entry point for standalone Whisper test.

    Usage:
        modal run modal_whisper.py
    """
    import os
    audio_file = "test_files/test_audio.mp3"

    if not os.path.exists(audio_file):
        print(f"❌ Error: Could not find {audio_file}!")
        print("Please put a small audio file in the test_files folder "
              "and name it 'test_audio.mp3' before running this.")
        return

    audio_bytes = open(audio_file, "rb").read()

    size_mb = len(audio_bytes) / (1024 * 1024)
    print(f"⬆️  Uploading to cloud H100 GPU (large-v3, v2): {size_mb:.2f} MB")

    # Send it to the H100 GPU
    text = transcribe.remote(audio_bytes)

    print(f"⬇️  Downloaded from cloud: {len(text)} chars of text")
    print(f"\nFinal Transcription:\n{text}")
