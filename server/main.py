"""
Ghost Sync — Server v3
=======================
FastAPI backend for subtitle transcription, streaming translation, and caching.

v3: Added streaming translation via SSE (Server-Sent Events).
    New endpoints:
      POST /translate_stream  — Streams translated SRT chunks live via SSE
      GET  /download_srt/{id} — Download a cached translated SRT file
"""

import os
import json
import tempfile
import shutil

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, Response

from server.sync_engine import sync_subtitle, audio_to_vad_signal, srt_to_presence_signal, find_offset, extract_audio_pcm
from server.identifier import identify_from_filename
from server.transcriber import transcribe_video  # fallback for simple /transcribe
from server.pipeline import run_pipeline          # v3 full pipeline (no chunking)
from server.translator import translate_srt, translate_srt_chunked
from server import db


app = FastAPI(
    title="Ghost Sync",
    description="Whisper-powered subtitle engine — transcribe once, cache forever, stream translations live.",
    version="0.4.0",
)


# =============================================================================
# Health Check
# =============================================================================

@app.get("/health")
def health_check():
    """Basic health check — verify server is running."""
    return {"status": "ok", "version": "0.4.0"}


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

    # Step 3: Full v3 pipeline (single GPU Whisper + sound detection)
    if not filepath or not os.path.exists(filepath):
        raise HTTPException(
            status_code=400,
            detail="File path required for transcription. Pass the full path to the video."
        )

    print(f"[Pipeline] Cache miss. Running v3 pipeline on '{info['title']}'...")
    pipeline_result = run_pipeline(filepath, language=language)

    # Step 4: Save to cache (never transcribe this movie again)
    sub_id = db.save_subtitle(
        movie_id=movie_id,
        srt_content=pipeline_result["srt_content"],
        language=pipeline_result["language"],
        source="pipeline-v3",
        download_count=0,
    )

    print(f"[Pipeline] v3 complete and cached! {pipeline_result['segment_count']} entries "
          f"({pipeline_result['dialogue_count']} dialogue + "
          f"{len(pipeline_result.get('sound_tags', []))} sound tags)")

    return {
        "source": "pipeline-v3",
        "movie_title": info["title"],
        "movie_year": info["year"],
        "srt_content": pipeline_result["srt_content"],
        "subtitle_id": sub_id,
        "language": pipeline_result["language"],
        "segment_count": pipeline_result["segment_count"],
        "dialogue_count": pipeline_result["dialogue_count"],
        "sound_tags": pipeline_result.get("sound_tags", []),
    }


# =============================================================================
# Streaming Translation via SSE (Server-Sent Events)
# =============================================================================

@app.post("/translate_stream")
def translate_stream(
    srt_content: str = Form(...),
    target_language: str = Form(...),
    movie_id: int = Form(None),
    source_language: str = Form(None),
):
    """Stream translated subtitle chunks in real-time via SSE.

    The client receives translated SRT chunks as Gemini completes them.
    First chunk arrives in ~8-12 seconds. Each chunk covers ~3-5 min of movie.

    SSE Event Format:
        data: {"chunk_index": 0, "total_chunks": 30, "srt_chunk": "1\\n00:00:01...", ...}

    Final event (is_final=true) triggers the client to show the Download button.
    After all chunks are sent, the full translated SRT is cached to the database.

    Args:
        srt_content:     The full original-language SRT string.
        target_language: Language to translate into (e.g., "Yoruba", "French").
        movie_id:        Optional movie ID for caching the completed translation.
        source_language: Optional source language code for cache metadata.
    """
    def event_generator():
        all_chunks = []  # collect for caching at the end

        for chunk_data in translate_srt_chunked(srt_content, target_language):
            # Collect the SRT text for stitching later
            if chunk_data.get("srt_chunk"):
                all_chunks.append(chunk_data["srt_chunk"])

            # Send this chunk to the client immediately
            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

        # All chunks done — stitch and cache the full translated SRT
        full_translated_srt = "\n\n".join(all_chunks)

        if movie_id and full_translated_srt:
            try:
                sub_id = db.save_subtitle(
                    movie_id=movie_id,
                    srt_content=full_translated_srt,
                    language=target_language.lower()[:3],  # normalize to short code
                    source="gemini-stream",
                    is_original=0,
                    translated_from=source_language,
                )
                # Send a cache confirmation event
                yield f"data: {json.dumps({'cached': True, 'subtitle_id': sub_id})}\n\n"
                print(f"💾 [Cache] Translated SRT saved (subtitle_id={sub_id}, "
                      f"lang={target_language}, movie_id={movie_id})")
            except Exception as e:
                print(f"⚠️  [Cache] Failed to save translated SRT: {e}")
                yield f"data: {json.dumps({'cached': False, 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering if behind reverse proxy
        },
    )


# =============================================================================
# Download Translated SRT
# =============================================================================

@app.get("/download_srt/{subtitle_id}")
def download_srt(subtitle_id: int):
    """Download a cached translated SRT file.

    Returns the SRT as a downloadable .srt file attachment.
    """
    conn = db.get_connection()
    row = conn.execute(
        "SELECT * FROM subtitles WHERE id = ?", (subtitle_id,)
    ).fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Subtitle not found")

    sub = dict(row)
    language = sub.get("language", "translated")
    filename = f"kalaye_{language}.srt"

    return Response(
        content=sub["srt_content"],
        media_type="application/x-subrip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


# =============================================================================
# Single-Shot Translation (non-streaming, for simple use)
# =============================================================================

@app.post("/translate")
def translate_endpoint(
    srt_content: str = Form(...),
    target_language: str = Form(...),
):
    """Translate a full SRT in one shot (non-streaming).

    For short SRTs or when the client doesn't support SSE.
    """
    translated = translate_srt(srt_content, target_language)
    return {
        "translated_srt": translated,
        "target_language": target_language,
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
