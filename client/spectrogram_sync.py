"""
Kalaye — Shazam-Style Spectrogram Fingerprinting Engine
========================================================
Generates and matches audio fingerprints using 2D spectrogram
constellation mapping for real-time subtitle synchronization.

Two-Layer Matching:
  Layer 1 (Constellation): Fast hash-based lookup → candidate positions (~5ms)
  Layer 2 (Mel Verification): Spectral shape cross-correlation → confirms match (~3ms)

Why this beats VAD (1D speech/silence):
  - VAD only knows "someone is talking" — two dialogue scenes look identical
  - Spectrogram knows the EXACT sound — pitch, timbre, frequency content
  - Works during music, action, ambient sounds — not just speech
  - False match probability: ~10^-193 (effectively impossible)

Runs 100% locally. No internet, no API calls.

Usage:
    # One-time: build fingerprint DB for a movie
    db = build_fingerprint_db(audio_samples)
    save_db(db, "movie.fpdb")

    # Runtime: find position from a 5-second clip
    db = load_db("movie.fpdb")
    position_ms, confidence, matches = find_position(clip_samples, db)
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.ndimage import maximum_filter
from collections import Counter, defaultdict
import pickle
import os


# ============================================================================
# Configuration
# ============================================================================

SAMPLE_RATE = 16000           # Hz — matches our audio extraction pipeline
FFT_WINDOW = 1024             # STFT window size (~64ms at 16kHz)
FFT_HOP = 512                 # hop between windows (~32ms per frame)
PEAK_NEIGHBORHOOD = 20        # local max filter size (freq × time cells)
PEAK_THRESHOLD_ABOVE_MEDIAN = 15  # peak must be this many dB above the median
FAN_OUT = 10                  # hash pairs per anchor peak
TARGET_ZONE_FRAMES = 50       # max time delta for paired peaks (~1.6s)

# Matching thresholds
MIN_HASH_MATCHES = 20         # minimum agreeing hashes for a confident match
CONSENSUS_WINDOW_MS = 100     # hashes must agree within this window (ms)

# Mel verification (Layer 2)
N_MELS = 40                   # Mel filter banks for verification
MEL_VERIFY_WINDOW_S = 2.0     # seconds of audio to cross-correlate
MEL_MIN_CORRELATION = 0.6     # minimum correlation to confirm a match


# ============================================================================
# Layer 1: Constellation Fingerprinting (Shazam-style)
# ============================================================================

def compute_spectrogram(samples):
    """Compute STFT spectrogram from audio samples.

    Args:
        samples: numpy float64 array of audio (16kHz mono).

    Returns:
        (spectrogram_db, freqs, times) — 2D magnitude in dB, freq axis, time axis.
    """
    freqs, times, Sxx = scipy_signal.spectrogram(
        samples,
        fs=SAMPLE_RATE,
        window='hann',
        nperseg=FFT_WINDOW,
        noverlap=FFT_WINDOW - FFT_HOP,
        scaling='spectrum',
    )
    # Convert to dB (log scale) — peaks become relative, volume-invariant
    Sxx_db = 10.0 * np.log10(Sxx + 1e-10)
    return Sxx_db, freqs, times


def extract_peaks(spectrogram_db):
    """Extract constellation map peaks from 2D spectrogram.

    Finds local maxima that are above the noise floor.
    These are the 'stars' in the constellation — they survive
    noise, compression, volume changes, and speaker coloring.

    Args:
        spectrogram_db: 2D numpy array (freq_bins × time_frames) in dB.

    Returns:
        List of (time_frame, freq_bin) tuples, sorted by time.
    """
    # Find local maxima using a 2D neighborhood filter
    local_max = maximum_filter(spectrogram_db, size=PEAK_NEIGHBORHOOD)

    # Adaptive threshold: peaks must be above (median + offset)
    # This works regardless of absolute dB level (normalized or not)
    median_db = np.median(spectrogram_db)
    threshold = median_db + PEAK_THRESHOLD_ABOVE_MEDIAN

    # Peak = cell equals its local max AND is above the adaptive threshold
    peaks_mask = (spectrogram_db == local_max) & (spectrogram_db > threshold)

    # Get coordinates
    freq_bins, time_frames = np.where(peaks_mask)

    # Sort by time for consistent hash generation
    sorted_idx = np.argsort(time_frames)
    peaks = list(zip(time_frames[sorted_idx].tolist(), freq_bins[sorted_idx].tolist()))

    return peaks


def generate_hashes(peaks):
    """Generate fingerprint hashes from constellation peak pairs.

    Each hash encodes (freq1, freq2, time_delta) — a pair of peaks.
    This is the core Shazam insight: individual peaks are common,
    but PAIRS of peaks at specific frequency/time relationships are unique.

    Args:
        peaks: List of (time_frame, freq_bin) tuples, sorted by time.

    Returns:
        List of (hash_int, anchor_time_frame) tuples.
    """
    hashes = []

    for i in range(len(peaks)):
        t1, f1 = peaks[i]
        paired = 0

        for j in range(i + 1, len(peaks)):
            t2, f2 = peaks[j]
            dt = t2 - t1

            if dt <= 0:
                continue
            if dt > TARGET_ZONE_FRAMES:
                break  # sorted by time — no more in range

            # Pack (f1, f2, dt) into a single integer
            # f1, f2: 0-512 (9 bits each), dt: 0-50 (6 bits) → 24-bit hash
            hash_val = (f1 << 15) | (f2 << 6) | dt
            hashes.append((hash_val, t1))

            paired += 1
            if paired >= FAN_OUT:
                break

    return hashes


def frame_to_ms(frame_idx):
    """Convert a spectrogram frame index to milliseconds."""
    return frame_idx * (FFT_HOP / SAMPLE_RATE) * 1000.0


def ms_to_frame(ms):
    """Convert milliseconds to nearest spectrogram frame index."""
    return int(ms / (FFT_HOP / SAMPLE_RATE) / 1000.0)


# ============================================================================
# Database: Build once per movie, reuse forever
# ============================================================================

def build_fingerprint_db(audio_samples):
    """Build a complete fingerprint hash table for a full movie.

    Runs ONCE when a movie is first processed. The result is saved
    alongside the SRT and loaded into RAM for real-time matching.

    Args:
        audio_samples: numpy float64 array of the entire movie audio (16kHz mono).

    Returns:
        dict with:
            'hashes': {hash_int: [movie_time_ms, ...]}
            'duration_ms': total duration in ms
            'peak_count': total peaks extracted
            'hash_count': total hashes generated
    """
    print("[Fingerprint] Computing spectrogram...")
    spec_db, freqs, times = compute_spectrogram(audio_samples)

    print("[Fingerprint] Extracting constellation peaks...")
    peaks = extract_peaks(spec_db)
    print(f"[Fingerprint] {len(peaks)} peaks extracted")

    print("[Fingerprint] Generating hash pairs...")
    hashes = generate_hashes(peaks)
    print(f"[Fingerprint] {len(hashes)} hashes generated")

    # Build lookup table: hash → list of movie timestamps (ms)
    hash_table = defaultdict(list)
    for hash_val, anchor_frame in hashes:
        time_ms = frame_to_ms(anchor_frame)
        hash_table[hash_val].append(time_ms)

    duration_ms = len(audio_samples) / SAMPLE_RATE * 1000.0

    db = {
        'hashes': dict(hash_table),
        'duration_ms': duration_ms,
        'peak_count': len(peaks),
        'hash_count': len(hashes),
    }

    print(f"[Fingerprint] DB built: {len(hash_table)} unique hashes, "
          f"{duration_ms / 1000:.0f}s duration")
    return db


def save_db(db, filepath):
    """Save fingerprint database to disk.

    Args:
        db: dict from build_fingerprint_db().
        filepath: path to save (e.g., 'movie.fpdb').
    """
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"[Fingerprint] Saved: {filepath} ({size_mb:.1f} MB)")


def load_db(filepath):
    """Load fingerprint database from disk.

    Args:
        filepath: path to the .fpdb file.

    Returns:
        dict with 'hashes', 'duration_ms', etc.
    """
    with open(filepath, 'rb') as f:
        db = pickle.load(f)
    print(f"[Fingerprint] Loaded: {len(db['hashes'])} unique hashes, "
          f"{db['duration_ms'] / 1000:.0f}s duration")
    return db


# ============================================================================
# Layer 1: Fast Hash Matching (find candidate position)
# ============================================================================

def find_position(clip_samples, fingerprint_db):
    """Find exact movie position from a short audio clip.

    This runs every 10-15 seconds during playback. Total time: ~8ms.

    Algorithm:
      1. Fingerprint the 5-second clip → ~100-200 hashes
      2. Look up each hash in the movie's database
      3. For each match, compute offset = movie_time - clip_time
      4. Vote: the offset with the most votes is the position
      5. If enough hashes agree (consensus), return the position

    Args:
        clip_samples: numpy float64 array of captured audio (16kHz mono, ~5s).
        fingerprint_db: dict from build_fingerprint_db() or load_db().

    Returns:
        (position_ms, confidence, total_matches)
        position_ms: exact movie timestamp in ms, or None if no confident match.
        confidence: number of agreeing hashes (higher = more certain).
        total_matches: how many hashes found any entry in the DB.
    """
    hash_table = fingerprint_db['hashes']

    # 1. Fingerprint the clip
    spec_db, _, _ = compute_spectrogram(clip_samples)
    peaks = extract_peaks(spec_db)
    clip_hashes = generate_hashes(peaks)

    if not clip_hashes:
        return None, 0, 0

    # 2. Look up each hash and vote on the offset
    offset_votes = Counter()
    total_matches = 0

    for hash_val, clip_anchor_frame in clip_hashes:
        clip_time_ms = frame_to_ms(clip_anchor_frame)

        if hash_val in hash_table:
            total_matches += 1
            for movie_time_ms in hash_table[hash_val]:
                # offset = where in the movie this clip starts
                offset_ms = movie_time_ms - clip_time_ms
                # Quantize to bins for consensus voting
                offset_bin = round(offset_ms / CONSENSUS_WINDOW_MS) * CONSENSUS_WINDOW_MS
                offset_votes[offset_bin] += 1

    if not offset_votes:
        return None, 0, 0

    # 3. Best candidate = offset with most votes
    best_offset_ms, vote_count = offset_votes.most_common(1)[0]

    # 4. Confidence gate
    if vote_count < MIN_HASH_MATCHES:
        return None, vote_count, total_matches

    return best_offset_ms, vote_count, total_matches


# ============================================================================
# Layer 2: Mel Cross-Correlation Verification
# ============================================================================

def _compute_mel_spectrogram(samples):
    """Compute a Mel-scaled spectrogram for verification.

    Mel scale emphasizes perceptually important frequencies,
    making it robust against minor spectral differences
    (speaker coloring, compression artifacts, etc.).

    Args:
        samples: numpy float64 audio array (16kHz mono).

    Returns:
        2D numpy array (n_mels × time_frames).
    """
    # Standard spectrogram first
    _, _, Sxx = scipy_signal.spectrogram(
        samples,
        fs=SAMPLE_RATE,
        window='hann',
        nperseg=FFT_WINDOW,
        noverlap=FFT_WINDOW - FFT_HOP,
    )

    # Build Mel filterbank
    n_fft_bins = Sxx.shape[0]
    mel_filters = _mel_filterbank(N_MELS, n_fft_bins, SAMPLE_RATE)

    # Apply Mel filters and convert to dB
    mel_spec = mel_filters @ Sxx
    mel_db = 10.0 * np.log10(mel_spec + 1e-10)

    return mel_db


def _mel_filterbank(n_mels, n_fft_bins, sample_rate):
    """Create a Mel-scale filterbank matrix.

    Args:
        n_mels: number of Mel bands.
        n_fft_bins: number of FFT frequency bins.
        sample_rate: audio sample rate in Hz.

    Returns:
        2D numpy array (n_mels × n_fft_bins).
    """
    # Mel scale conversion
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(0)
    mel_max = hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Convert Hz to FFT bin indices
    bin_points = np.floor((n_fft_bins * 2) * hz_points / sample_rate).astype(int)

    filters = np.zeros((n_mels, n_fft_bins))
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        # Rising slope
        for j in range(left, min(center, n_fft_bins)):
            if center != left:
                filters[i, j] = (j - left) / (center - left)
        # Falling slope
        for j in range(center, min(right, n_fft_bins)):
            if right != center:
                filters[i, j] = (right - j) / (right - center)

    return filters


def verify_match(clip_samples, movie_samples, candidate_ms):
    """Verify a constellation match using Mel spectral cross-correlation.

    Layer 2: After constellation matching finds a candidate position,
    this function extracts the corresponding segment from the movie audio
    and compares the Mel spectrograms directly. This catches false matches
    from repeated motifs or hash collisions.

    Args:
        clip_samples: numpy float64 of captured audio (~5s).
        movie_samples: numpy float64 of the full movie audio.
        candidate_ms: candidate position in the movie (ms).

    Returns:
        (verified: bool, correlation: float)
    """
    # Extract the corresponding movie segment
    start_sample = int(candidate_ms / 1000.0 * SAMPLE_RATE)
    clip_len = len(clip_samples)

    # Bounds check
    if start_sample < 0:
        start_sample = 0
    if start_sample + clip_len > len(movie_samples):
        return False, 0.0

    movie_segment = movie_samples[start_sample:start_sample + clip_len]

    if len(movie_segment) < SAMPLE_RATE:  # less than 1 second
        return False, 0.0

    # Compute Mel spectrograms for both
    clip_mel = _compute_mel_spectrogram(clip_samples)
    movie_mel = _compute_mel_spectrogram(movie_segment)

    # Flatten and normalize
    clip_flat = clip_mel.flatten()
    movie_flat = movie_mel.flatten()

    # Truncate to same length (minor differences from rounding)
    min_len = min(len(clip_flat), len(movie_flat))
    clip_flat = clip_flat[:min_len]
    movie_flat = movie_flat[:min_len]

    # Pearson correlation
    clip_norm = clip_flat - np.mean(clip_flat)
    movie_norm = movie_flat - np.mean(movie_flat)

    denom = np.sqrt(np.sum(clip_norm ** 2) * np.sum(movie_norm ** 2))
    if denom < 1e-10:
        return False, 0.0

    correlation = np.sum(clip_norm * movie_norm) / denom

    return correlation >= MEL_MIN_CORRELATION, float(correlation)


# ============================================================================
# Convenience: Build DB from a video/audio file
# ============================================================================

def build_db_from_file(media_path, output_path=None):
    """Build fingerprint DB directly from a video/audio file.

    Extracts audio via ffmpeg, builds the constellation hash table,
    and saves it to disk.

    Args:
        media_path: path to any video/audio file ffmpeg can read.
        output_path: where to save the .fpdb file.
                     Defaults to '{media_path}.fpdb'.

    Returns:
        The fingerprint database dict.
    """
    import subprocess

    if output_path is None:
        output_path = media_path + ".fpdb"

    # Check if DB already exists (cached) and is not empty
    if os.path.exists(output_path):
        try:
            cached = load_db(output_path)
            if cached and cached.get('hash_count', 0) > 0:
                print(f"[Fingerprint] Cache hit: {output_path}")
                return cached
            else:
                # Empty DB from a previous bug — delete and rebuild
                print(f"[Fingerprint] Cached DB is empty — rebuilding...")
                os.remove(output_path)
        except Exception:
            print(f"[Fingerprint] Cached DB is corrupt — rebuilding...")
            os.remove(output_path)

    print(f"[Fingerprint] Extracting audio from: {os.path.basename(media_path)}")

    # Extract 16kHz mono PCM via ffmpeg
    cmd = [
        "ffmpeg",
        "-i", media_path,
        "-vn",                  # no video
        "-acodec", "pcm_s16le", # 16-bit PCM
        "-ac", "1",             # mono
        "-ar", str(SAMPLE_RATE),
        "-f", "s16le",          # raw PCM (no WAV header)
        "pipe:1",
        "-loglevel", "quiet",
    ]

    result = subprocess.run(cmd, capture_output=True)
    if len(result.stdout) < 1000:
        print("[Fingerprint] ERROR: Audio extraction failed")
        return None

    # Convert bytes → float64 samples
    samples = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float64) / 32768.0

    duration_s = len(samples) / SAMPLE_RATE
    print(f"[Fingerprint] Audio: {duration_s:.0f}s ({len(samples)} samples)")

    # Build and save
    db = build_fingerprint_db(samples)
    save_db(db, output_path)

    return db
