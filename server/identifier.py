"""
Ghost Sync — Movie Identifier
===============================
Identifies a movie from a video filename.
Extracts title, year, and quality info from common naming patterns.

Future: ACRCloud audio fingerprinting + Whisper ASR fallback.
"""

import re
import hashlib


# Common patterns in media filenames
QUALITY_TAGS = [
    "2160p", "1080p", "720p", "480p", "360p",
    "4K", "UHD", "FHD", "HD",
    "BluRay", "Blu-Ray", "BDRip", "BRRip",
    "WEBRip", "WEB-DL", "WEBDL", "WEB",
    "HDRip", "DVDRip", "DVDScr",
    "CAM", "HDCAM", "TS", "HDTS", "TC",
    "AMZN", "NF", "HMAX", "DSNP", "ATVP",
]

CODEC_TAGS = [
    "x264", "x265", "H264", "H265", "HEVC", "AVC",
    "XviD", "DivX", "VP9", "AV1",
    "AAC", "AC3", "DTS", "FLAC", "MP3",
    "10bit", "HDR", "SDR", "Atmos",
]

# Build regex pattern to strip these tags
_strip_pattern = re.compile(
    r"(?:" + "|".join(re.escape(t) for t in QUALITY_TAGS + CODEC_TAGS) + r")",
    re.IGNORECASE
)


def parse_filename(filename):
    """
    Extract movie title and year from a filename.

    Examples:
        'Interstellar.2014.BluRay.1080p.mkv' → ('Interstellar', 2014)
        'The Matrix (1999) [720p]' → ('The Matrix', 1999)
        'movie.mp4' → ('movie', None)

    Returns:
        dict with keys: title, year, original_filename, filename_hash
    """
    # Remove file extension
    name = re.sub(r"\.[a-zA-Z0-9]{2,4}$", "", filename)

    # Try to find a year (1920-2029)
    year_match = re.search(r"[\.\s\(\[_-]((?:19|20)\d{2})[\.\s\)\]_-]", name)
    year = int(year_match.group(1)) if year_match else None

    # Cut everything after the year (usually quality/codec tags)
    if year_match:
        name = name[:year_match.start()]

    # Strip known quality/codec tags from what remains
    name = _strip_pattern.sub("", name)

    # Replace dots, underscores, dashes with spaces
    name = re.sub(r"[\.\-_\[\]\(\)]", " ", name)

    # Clean up whitespace
    title = re.sub(r"\s+", " ", name).strip()

    # Generate a hash for cache lookups
    filename_hash = hashlib.md5(filename.lower().encode()).hexdigest()

    return {
        "title": title,
        "year": year,
        "original_filename": filename,
        "filename_hash": filename_hash,
    }


def identify_from_filename(filename):
    """
    Pre-check ladder — Step 1: Filename parsing.
    Returns parsed movie info or None if filename is too generic.
    """
    info = parse_filename(filename)

    # If the title is too short or generic, skip
    if len(info["title"]) < 2:
        return None

    return info


# =============================================================================
# Future: ACRCloud + Whisper (stubs)
# =============================================================================

def identify_from_audio_fingerprint(audio_bytes):
    """
    Pre-check ladder — Step 2: ACRCloud audio fingerprinting.
    TODO: Implement when API key is available.
    """
    raise NotImplementedError("ACRCloud integration not yet implemented")


def identify_from_whisper(audio_bytes):
    """
    Pre-check ladder — Step 3: Whisper ASR fallback.
    TODO: Implement when OpenAI API key is available.
    """
    raise NotImplementedError("Whisper integration not yet implemented")
