"""
Ghost Sync — API Client
=========================
HTTP client for the desktop app to communicate with the Ghost Sync server.

v2: Added stream_translation() for SSE streaming translations.
"""

import requests
import json
import os


DEFAULT_SERVER = "http://localhost:8000"


class GhostSyncClient:
    """Client to talk to the Ghost Sync server."""

    def __init__(self, server_url=None):
        self.server_url = server_url or os.getenv("GHOST_SYNC_SERVER", DEFAULT_SERVER)

    def health_check(self):
        """Check if the server is running."""
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=5)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False

    def identify(self, filename):
        """
        Identify a movie from its filename.
        Returns movie info dict or None.
        """
        resp = requests.post(
            f"{self.server_url}/identify",
            data={"filename": filename},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
        return None

    def sync(self, video_path, srt_path):
        """
        Upload a video and SRT to get the sync offset.
        Returns dict with offset_seconds and confidence.
        """
        with open(video_path, "rb") as video_file, open(srt_path, "rb") as srt_file:
            resp = requests.post(
                f"{self.server_url}/sync",
                files={
                    "video": (os.path.basename(video_path), video_file),
                    "srt": (os.path.basename(srt_path), srt_file),
                },
                timeout=120,  # sync can take a while for long videos
            )

        if resp.status_code == 200:
            return resp.json()
        else:
            raise Exception(f"Sync failed: {resp.status_code} — {resp.text}")

    def get_subtitles(self, movie_id, language="en"):
        """Get available subtitles for a movie."""
        resp = requests.get(
            f"{self.server_url}/subtitles/{movie_id}",
            params={"language": language},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
        return None

    def pipeline(self, filename, language=None, filepath=None):
        """
        Full pipeline: check cache → Whisper transcribe → return SRT.
        Pass filepath so server can transcribe the audio.
        """
        data = {"filename": filename, "language": language}
        if filepath:
            data["filepath"] = filepath

        resp = requests.post(
            f"{self.server_url}/pipeline",
            data=data,
            timeout=900,  # Whisper transcription can take ~10 min on CPU
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 404:
            return None
        else:
            raise Exception(f"Pipeline failed: {resp.status_code} — {resp.text}")

    # =========================================================================
    # Streaming Translation (SSE)
    # =========================================================================

    def stream_translation(self, srt_content, target_language,
                           movie_id=None, source_language=None,
                           on_chunk=None, on_error=None, on_complete=None):
        """Stream translated SRT chunks from the server via SSE.

        Connects to POST /translate_stream and yields translated chunks
        as they arrive. Each chunk covers ~50 subtitle blocks (~3-5 min).

        Args:
            srt_content:     Full original-language SRT string.
            target_language: Language to translate into (e.g., "Yoruba").
            movie_id:        Optional movie ID for server-side caching.
            source_language: Optional source language code.
            on_chunk:        Callback(dict) called for each translated chunk.
            on_error:        Callback(str) called on errors.
            on_complete:     Callback(str, int|None) called when done with
                             (full_srt, subtitle_id).

        Returns:
            Full translated SRT string (all chunks stitched together).
        """
        data = {
            "srt_content": srt_content,
            "target_language": target_language,
        }
        if movie_id is not None:
            data["movie_id"] = movie_id
        if source_language:
            data["source_language"] = source_language

        all_chunks = []
        subtitle_id = None

        try:
            resp = requests.post(
                f"{self.server_url}/translate_stream",
                data=data,
                stream=True,
                timeout=600,  # 10 min total timeout
            )

            if resp.status_code != 200:
                error_msg = f"Stream failed: {resp.status_code} — {resp.text[:200]}"
                if on_error:
                    on_error(error_msg)
                raise Exception(error_msg)

            # Parse SSE events
            buffer = ""
            for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    buffer += chunk

                    # Process complete SSE events (each ends with \n\n)
                    while "\n\n" in buffer:
                        event_str, buffer = buffer.split("\n\n", 1)
                        event_str = event_str.strip()

                        if not event_str.startswith("data: "):
                            continue

                        json_str = event_str[6:]  # strip "data: "
                        try:
                            event_data = json.loads(json_str)
                        except json.JSONDecodeError:
                            continue

                        # Cache confirmation event (final)
                        if "cached" in event_data:
                            subtitle_id = event_data.get("subtitle_id")
                            continue

                        # Error in a chunk
                        if "error" in event_data:
                            if on_error:
                                on_error(f"Chunk {event_data.get('chunk_index', '?')}: "
                                         f"{event_data['error']}")
                            continue

                        # Regular translated chunk
                        srt_chunk = event_data.get("srt_chunk", "")
                        if srt_chunk:
                            all_chunks.append(srt_chunk)

                        if on_chunk:
                            on_chunk(event_data)

        except requests.ConnectionError as e:
            error_msg = f"Connection to server failed: {e}"
            if on_error:
                on_error(error_msg)
            raise

        # Stitch all chunks into one complete SRT
        full_srt = "\n\n".join(all_chunks)

        if on_complete:
            on_complete(full_srt, subtitle_id)

        return full_srt

    def download_srt(self, subtitle_id, save_path):
        """Download a cached translated SRT file from the server.

        Args:
            subtitle_id: ID from the cache confirmation event.
            save_path:   Local path to save the .srt file.
        """
        resp = requests.get(
            f"{self.server_url}/download_srt/{subtitle_id}",
            timeout=30,
        )
        if resp.status_code == 200:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(resp.text)
            return True
        else:
            raise Exception(f"Download failed: {resp.status_code}")
