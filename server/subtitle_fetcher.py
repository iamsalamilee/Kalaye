"""
Ghost Sync — Subtitle Fetcher
================================
Fetches subtitle files from OpenSubtitles API.
Searches by file hash (accurate) or movie name (fallback).

API Docs: https://opensubtitles.stoplight.io/docs/opensubtitles-api
Free tier: 200 requests/day, 40 requests/minute.
"""

import os
import struct
import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE = "https://api.opensubtitles.com/api/v1"
API_KEY = os.getenv("OPENSUBTITLES_API_KEY", "")
USERNAME = os.getenv("OPENSUBTITLES_USERNAME", "")
PASSWORD = os.getenv("OPENSUBTITLES_PASSWORD", "")


def _headers(token=None):
    """Build request headers with API key and optional auth token."""
    h = {
        "Api-Key": API_KEY,
        "Content-Type": "application/json",
        "User-Agent": "GhostSync v0.2",
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


# =============================================================================
# OpenSubtitles File Hash
# =============================================================================

def compute_file_hash(filepath):
    """
    Compute the OpenSubtitles hash for a video file.
    
    Algorithm: hash = file_size + first_64KB_checksum + last_64KB_checksum
    This uniquely identifies a video file regardless of filename.
    
    Returns: (hash_hex, file_size)
    """
    BLOCK_SIZE = 65536  # 64KB

    file_size = os.path.getsize(filepath)
    if file_size < BLOCK_SIZE * 2:
        print(f"[SubFetcher] File too small for hash ({file_size} bytes)")
        return None, file_size

    file_hash = file_size

    with open(filepath, "rb") as f:
        # Read first 64KB
        for _ in range(BLOCK_SIZE // 8):
            buf = f.read(8)
            (val,) = struct.unpack("<Q", buf)
            file_hash += val
            file_hash &= 0xFFFFFFFFFFFFFFFF  # keep as 64-bit

        # Read last 64KB
        f.seek(-BLOCK_SIZE, 2)  # seek from end
        for _ in range(BLOCK_SIZE // 8):
            buf = f.read(8)
            (val,) = struct.unpack("<Q", buf)
            file_hash += val
            file_hash &= 0xFFFFFFFFFFFFFFFF

    hash_hex = f"{file_hash:016x}"
    print(f"[SubFetcher] File hash: {hash_hex} ({file_size / (1024*1024):.1f} MB)")
    return hash_hex, file_size


# =============================================================================
# Search Methods
# =============================================================================

def search_by_hash(filepath, language="en", limit=3):
    """
    Search OpenSubtitles by file hash — most accurate method.
    Matches the exact video file, not just the name.
    """
    if not API_KEY:
        return []

    file_hash, file_size = compute_file_hash(filepath)
    if not file_hash:
        return []

    params = {
        "moviehash": file_hash,
        "languages": language,
        "order_by": "download_count",
        "order_direction": "desc",
    }

    resp = requests.get(
        f"{API_BASE}/subtitles",
        params=params,
        headers=_headers(),
        timeout=15,
    )

    if resp.status_code != 200:
        print(f"[SubFetcher] Hash search failed: {resp.status_code}")
        return []

    return _parse_results(resp.json(), limit, "hash")


def search_subtitles(title, year=None, language="en", limit=3):
    """
    Search for subtitles by movie title (fallback method).
    """
    if not API_KEY:
        print("[SubFetcher] Error: No API key. Set OPENSUBTITLES_API_KEY in .env")
        return []

    params = {
        "query": title,
        "languages": language,
        "order_by": "download_count",
        "order_direction": "desc",
    }
    if year:
        params["year"] = year

    resp = requests.get(
        f"{API_BASE}/subtitles",
        params=params,
        headers=_headers(),
        timeout=15,
    )

    if resp.status_code != 200:
        print(f"[SubFetcher] Title search failed: {resp.status_code}")
        return []

    return _parse_results(resp.json(), limit, "title")


def _parse_results(data, limit, method):
    """Parse API response into clean subtitle list."""
    results = data.get("data", [])
    subtitles = []

    for item in results[:limit]:
        attrs = item.get("attributes", {})
        files = attrs.get("files", [])
        if not files:
            continue

        subtitles.append({
            "id": item.get("id"),
            "file_id": files[0].get("file_id"),
            "filename": files[0].get("file_name", "unknown.srt"),
            "download_count": attrs.get("download_count", 0),
            "language": attrs.get("language", "en"),
            "fps": attrs.get("fps", 0),
            "release": attrs.get("release", ""),
            "from_trusted": attrs.get("from_trusted", False),
            "match_method": method,
        })

    method_label = "hash match" if method == "hash" else "title search"
    print(f"[SubFetcher] Found {len(subtitles)} subtitle(s) via {method_label}")
    for i, s in enumerate(subtitles):
        trust = " trusted" if s["from_trusted"] else ""
        print(f"  {i+1}. {s['filename']} ({s['download_count']} downloads){trust}")

    return subtitles


# =============================================================================
# Login & Download
# =============================================================================

def login():
    """Authenticate with OpenSubtitles to get a download token."""
    if not USERNAME or not PASSWORD:
        print("[SubFetcher] Warning: No OpenSubtitles credentials in .env")
        return None

    resp = requests.post(
        f"{API_BASE}/login",
        json={"username": USERNAME, "password": PASSWORD},
        headers=_headers(),
        timeout=10,
    )

    if resp.status_code == 200:
        token = resp.json().get("token")
        print(f"[SubFetcher] Logged in successfully")
        return token
    else:
        print(f"[SubFetcher] Login failed: {resp.status_code} — {resp.text}")
        return None


def download_subtitle(file_id, token=None):
    """Download a subtitle file by its file_id."""
    if not token:
        token = login()
    if not token:
        return None

    resp = requests.post(
        f"{API_BASE}/download",
        json={"file_id": file_id},
        headers=_headers(token),
        timeout=15,
    )

    if resp.status_code != 200:
        print(f"[SubFetcher] Download failed: {resp.status_code} — {resp.text}")
        return None

    download_url = resp.json().get("link")
    if not download_url:
        return None

    file_resp = requests.get(download_url, timeout=30)
    if file_resp.status_code == 200:
        print(f"[SubFetcher] Downloaded subtitle ({len(file_resp.content)} bytes)")
        return file_resp.text
    return None


# =============================================================================
# High-Level API
# =============================================================================

def fetch_best_subtitle(title, year=None, language="en", filepath=None):
    """
    Smart subtitle fetch with fallback ladder:
    1. Try hash search (exact file match) — if filepath provided
    2. Fall back to title search

    Returns dict with srt_content, filename, download_count, match_method
    """
    results = []

    # Method 1: Hash search (most accurate)
    if filepath and os.path.exists(filepath):
        print(f"[SubFetcher] Trying hash search first...")
        results = search_by_hash(filepath, language=language)

    # Method 2: Title search (fallback)
    if not results:
        print(f"[SubFetcher] Trying title search: '{title}'")
        results = search_subtitles(title, year=year, language=language)

    if not results:
        return None

    # Login and download
    token = login()
    if not token:
        return None

    for sub in results:
        content = download_subtitle(sub["file_id"], token=token)
        if content:
            return {
                "srt_content": content,
                "filename": sub["filename"],
                "download_count": sub["download_count"],
                "file_id": sub["file_id"],
                "match_method": sub["match_method"],
            }

    return None


# =============================================================================
# Quick test
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  By title:  python -m server.subtitle_fetcher 'The Matrix' 1999")
        print("  By file:   python -m server.subtitle_fetcher --file movie.mp4")
        sys.exit(1)

    if sys.argv[1] == "--file":
        filepath = sys.argv[2]
        print(f"\nSearching by file hash: {filepath}")
        results = search_by_hash(filepath)
        if not results:
            print("No hash match. Trying filename parse...")
            from server.identifier import parse_filename
            info = parse_filename(os.path.basename(filepath))
            if info:
                results = search_subtitles(info["title"], year=info["year"])
    else:
        title = sys.argv[1]
        year = int(sys.argv[2]) if len(sys.argv) > 2 else None
        results = search_subtitles(title, year=year)

    if results:
        print(f"\nTop result: {results[0]['filename']} (via {results[0]['match_method']})")
