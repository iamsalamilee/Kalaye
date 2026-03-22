"""
Ghost Sync — Server
====================
FastAPI backend for subtitle sync, identification, and fetching.
"""

import os
import tempfile
import shutil

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from server.sync_engine import sync_subtitle, audio_to_vad_signal, srt_to_presence_signal, find_offset, extract_audio_pcm
from server.identifier import identify_from_filename, parse_filename
from server import db


app = FastAPI(
    title="Ghost Sync",
    description="Subtitle sync engine — identifies movies, fetches subtitles, syncs them.",
    version="0.1.0",
)


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health")
def health_check():
    """Basic health check — verify server is running."""
    return {"status": "ok", "version": "0.1.0"}


# =============================================================================
# Identify a movie from its filename
# =============================================================================

@app.post("/identify")
def identify_movie(filename: str = Form(...)):
    """
    Identify a movie from its video filename.
    Returns parsed title, year, and whether it's in our database.
    """
    info = identify_from_filename(filename)
    if info is None:
        raise HTTPException(status_code=400, detail="Could not parse filename")

    # Check if we've seen this file before
    cached = db.find_movie_by_hash(info["filename_hash"])
    if cached:
        return {
            "match": True,
            "movie_id": cached["id"],
            "title": cached["title"],
            "year": cached["year"],
            "source": "cache",
        }

    # New movie — save it
    movie_id = db.create_movie(
        title=info["title"],
        year=info["year"],
        filename_hash=info["filename_hash"],
    )

    return {
        "match": True,
        "movie_id": movie_id,
        "title": info["title"],
        "year": info["year"],
        "source": "filename_parse",
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
    # Save uploaded files to temp directory
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

        # Run the sync algorithm
        result = sync_subtitle(video_path, srt_path)

        return {
            "offset_seconds": result["offset"],
            "confidence": result["confidence"],
            "recommendation": f"Shift subtitles by {result['offset']:+.3f} seconds",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temp files
        shutil.rmtree(tmp_dir, ignore_errors=True)


# =============================================================================
# Get subtitles for a movie
# =============================================================================

@app.get("/subtitles/{movie_id}")
def get_subtitles(movie_id: int, language: str = "en"):
    """
    Get all stored subtitles for a movie.
    Returns list of available SRTs ordered by download count.
    """
    subs = db.get_subtitles_for_movie(movie_id, language)
    if not subs:
        raise HTTPException(status_code=404, detail="No subtitles found for this movie")

    return {
        "movie_id": movie_id,
        "language": language,
        "count": len(subs),
        "subtitles": [
            {
                "id": s["id"],
                "source": s["source"],
                "download_count": s["download_count"],
            }
            for s in subs
        ],
    }
