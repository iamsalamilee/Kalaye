"""
Ghost Sync — API Client
=========================
HTTP client for the desktop app to communicate with the Ghost Sync server.
"""

import requests
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
