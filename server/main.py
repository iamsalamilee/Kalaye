"""
Ghost Sync — Server
====================
FastAPI backend for subtitle transcription and caching.
Whisper-powered: transcribe once, cache forever, translate on demand.
"""

import os
import tempfile
import shutil

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from server.sync_engine import sync_subtitle, audio_to_vad_signal, srt_to_presence_signal, find_offset, extract_audio_pcm
from server.identifier import identify_from_filename
from server.transcriber import transcribe_video
from server import db


app = FastAPI(
    title="Ghost Sync",
    description="Whisper-powered subtitle engine — transcribe once, cache forever.",
    version="0.3.0",
)


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health")
def health_check():
    """Basic health check — verify server is running."""
    return {"status": "ok", "version": "0.3.0"}


# =============================================================================
# Main Pipeline: cache check → Whisper transcribe → save → return
# =============================================================================

@app.post("/pipeline")
def full_pipeline(
    filename: str = Form(...),
    language: str = Form(None),  # None = auto-detect original language
    filepath: str = Form(None),
):
    """
    Ghost Sync pipeline:
    1. Compute file hash → check cache
    2. Cache hit? → return cached subtitles instantly
    3. Cache miss? → Whisper transcribes audio → save to cache → return

    Once a movie is transcribed, it's cached forever. Never transcribes twice.
    """
    # Step 1: Identify the file
    info = identify_from_filename(filename)
    if info is None:
        raise HTTPException(status_code=400, detail="Could not parse movie from filename")

    # Step 2: Check cache
    cached = db.find_movie_by_hash(info["filename_hash"])
    if cached:
        subs = db.get_subtitles_for_movie(cached["id"], language)
        if subs:
            print(f"[Pipeline] Cache hit! '{cached['title']}' — returning instantly")
            return {
                "source": "cache",
                "movie_title": cached["title"],
                "movie_year": cached["year"],
                "srt_content": subs[0]["srt_content"],
                "subtitle_id": subs[0]["id"],
                "language": language,
            }
        movie_id = cached["id"]
    else:
        movie_id = db.create_movie(
            title=info["title"],
            year=info["year"],
            filename_hash=info["filename_hash"],
        )

    # Step 3: Whisper transcription
    if not filepath or not os.path.exists(filepath):
        raise HTTPException(
            status_code=400,
            detail="File path required for transcription. Pass the full path to the video."
        )

    print(f"[Pipeline] Cache miss. Transcribing '{info['title']}' with Whisper...")
    whisper_result = transcribe_video(filepath, language=language)

    # Step 4: Save to cache (never transcribe this movie again)
    sub_id = db.save_subtitle(
        movie_id=movie_id,
        srt_content=whisper_result["srt_content"],
        language=whisper_result["language"],
        source="whisper",
        download_count=0,
    )

    print(f"[Pipeline] Transcribed and cached! {whisper_result['segment_count']} segments")

    return {
        "source": "whisper",
        "movie_title": info["title"],
        "movie_year": info["year"],
        "srt_content": whisper_result["srt_content"],
        "subtitle_id": sub_id,
        "language": whisper_result["language"],
        "segment_count": whisper_result["segment_count"],
    }


# =============================================================================
# Direct Whisper transcription (no cache, no identification)
# =============================================================================

@app.post("/transcribe")
def transcribe_endpoint(filepath: str = Form(...), language: str = Form(None)):
    """
    Transcribe a video/audio file directly with Whisper.
    Returns SRT content. Does not save to cache.
    """
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"File not found: {filepath}")

    result = transcribe_video(filepath, language=language)

    return {
        "source": "whisper",
        "srt_content": result["srt_content"],
        "language": result["language"],
        "segment_count": result["segment_count"],
    }


# =============================================================================
# Sync: upload video + SRT, get the time offset
# =============================================================================

@app.post("/sync")
async def sync_files(
    video: UploadFile = File(...),
    srt: UploadFile = File(...),
):
    """
    Upload a video file and an SRT file.
    Returns the detected time offset and confidence.
    """
    tmp_dir = tempfile.mkdtemp(prefix="ghost_sync_")

    try:
        video_path = os.path.join(tmp_dir, video.filename)
        srt_path = os.path.join(tmp_dir, srt.filename)

        with open(video_path, "wb") as f:
            content = await video.read()
            f.write(content)

        with open(srt_path, "wb") as f:
            content = await srt.read()
            f.write(content)

        result = sync_subtitle(video_path, srt_path)

        return {
            "offset_seconds": result["offset"],
            "confidence": result["confidence"],
            "recommendation": f"Shift subtitles by {result['offset']:+.3f} seconds",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# Get cached subtitles for a movie
# =============================================================================

@app.get("/subtitles/{movie_id}")
def get_subtitles(movie_id: int, language: str = "en"):
    """Get cached subtitles for a movie by its ID."""
    subs = db.get_subtitles_for_movie(movie_id, language)
    if not subs:
        raise HTTPException(status_code=404, detail="No subtitles cached for this movie")

    return {
        "movie_id": movie_id,
        "language": language,
        "count": len(subs),
        "subtitles": [
            {
                "id": s["id"],
                "source": s["source"],
                "language": s["language"],
            }
            for s in subs
        ],
    }
