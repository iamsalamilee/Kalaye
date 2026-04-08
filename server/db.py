"""
Ghost Sync — Database Layer
============================
SQLite for local development. Can swap to Postgres for production.
"""

import sqlite3
import os
from datetime import datetime


DB_PATH = os.path.join(os.path.dirname(__file__), "ghost_sync.db")


def get_connection():
    """Get a database connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # better concurrent access
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS movies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            year INTEGER,
            filename_hash TEXT UNIQUE,
            acr_id TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS subtitles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            movie_id INTEGER NOT NULL,
            language TEXT NOT NULL DEFAULT 'und',
            source TEXT DEFAULT 'whisper',
            is_original INTEGER DEFAULT 1,        -- 1 = Whisper transcribed, 0 = LLM translated
            translated_from TEXT DEFAULT NULL,     -- e.g. 'ko' if this was translated from Korean
            srt_content TEXT NOT NULL,
            download_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (movie_id) REFERENCES movies(id),
            UNIQUE(movie_id, language)             -- one subtitle per language per movie
        );

        CREATE TABLE IF NOT EXISTS sync_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            movie_id INTEGER NOT NULL,
            subtitle_id INTEGER NOT NULL,
            offset_seconds REAL NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (movie_id) REFERENCES movies(id),
            FOREIGN KEY (subtitle_id) REFERENCES subtitles(id)
        );

        CREATE INDEX IF NOT EXISTS idx_movies_hash ON movies(filename_hash);
        CREATE INDEX IF NOT EXISTS idx_subtitles_movie ON subtitles(movie_id);
        CREATE INDEX IF NOT EXISTS idx_subtitles_lang ON subtitles(movie_id, language);
        CREATE INDEX IF NOT EXISTS idx_sync_movie ON sync_results(movie_id);
    """)

    # Migrate existing DBs — add new columns if they don't exist yet
    for col, definition in [
        ("is_original", "INTEGER DEFAULT 1"),
        ("translated_from", "TEXT DEFAULT NULL"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE subtitles ADD COLUMN {col} {definition}")
        except Exception:
            pass  # column already exists


    conn.commit()
    conn.close()


# =============================================================================
# Movie operations
# =============================================================================

def find_movie_by_hash(filename_hash):
    """Look up a movie by its filename hash."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM movies WHERE filename_hash = ?", (filename_hash,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def create_movie(title, year=None, filename_hash=None, acr_id=None):
    """Insert a new movie record."""
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO movies (title, year, filename_hash, acr_id) VALUES (?, ?, ?, ?)",
        (title, year, filename_hash, acr_id)
    )
    movie_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return movie_id


# =============================================================================
# Subtitle operations
# =============================================================================

def save_subtitle(movie_id, srt_content, language="und", source="whisper",
                  is_original=1, translated_from=None, download_count=0):
    """Save a subtitle for a movie. Upserts on (movie_id, language)."""
    conn = get_connection()
    cursor = conn.execute(
        """INSERT INTO subtitles
               (movie_id, language, source, is_original, translated_from, srt_content, download_count)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(movie_id, language) DO UPDATE SET
               srt_content=excluded.srt_content,
               source=excluded.source,
               is_original=excluded.is_original,
               translated_from=excluded.translated_from""",
        (movie_id, language, source, is_original, translated_from, srt_content, download_count)
    )
    sub_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return sub_id


def get_subtitles_for_movie(movie_id, language=None):
    """Get subtitles for a movie.
    - language specified: return only that language
    - language=None: return the best available (whatever Whisper detected)
    """
    conn = get_connection()
    if language:
        rows = conn.execute(
            """SELECT * FROM subtitles
               WHERE movie_id = ? AND language = ?
               ORDER BY download_count DESC""",
            (movie_id, language)
        ).fetchall()
    else:
        # Return best single subtitle in any language (Whisper auto-detected)
        rows = conn.execute(
            """SELECT * FROM subtitles
               WHERE movie_id = ?
               ORDER BY download_count DESC
               LIMIT 1""",
            (movie_id,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def find_movies_by_title(title, limit=5):
    """Find movies with similar titles — used for fingerprint-based disambiguation.
    Returns up to `limit` candidates ordered by creation date.
    """
    conn = get_connection()
    rows = conn.execute(
        """SELECT * FROM movies
           WHERE LOWER(title) LIKE LOWER(?)
           ORDER BY created_at DESC
           LIMIT ?""",
        (f"%{title}%", limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# =============================================================================
# Sync result operations
# =============================================================================

def save_sync_result(movie_id, subtitle_id, offset_seconds, confidence):
    """Save a sync result."""
    conn = get_connection()
    conn.execute(
        """INSERT INTO sync_results (movie_id, subtitle_id, offset_seconds, confidence)
           VALUES (?, ?, ?, ?)""",
        (movie_id, subtitle_id, offset_seconds, confidence)
    )
    conn.commit()
    conn.close()


def get_cached_sync(movie_id, subtitle_id):
    """Check if we already have a sync result for this movie+subtitle pair."""
    conn = get_connection()
    row = conn.execute(
        """SELECT * FROM sync_results
           WHERE movie_id = ? AND subtitle_id = ?
           ORDER BY created_at DESC LIMIT 1""",
        (movie_id, subtitle_id)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# Initialize database on import
init_db()
