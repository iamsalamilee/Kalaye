"""
Kalaye — Subtitle Translator (v2: Streaming Chunked)
=====================================================
Translates SRT subtitles via Gemini 2.5 Flash.

Two modes:
  1. translate_srt()          — Single-shot: send full SRT, wait, get full SRT back.
                                 Good for short files (<100 blocks) or background jobs.

  2. translate_srt_chunked()  — Generator: yields translated chunks of ~50 blocks each.
                                 First chunk arrives in ~8-12s. Each subsequent chunk
                                 includes 3-line overlap context from the previous one
                                 so Gemini maintains narrative coherence.
                                 Used by the /translate_stream SSE endpoint.
"""

import os
import re
import json
import time
import requests
from dotenv import load_dotenv

# Load .env file at the start of the module
load_dotenv()


# =============================================================================
# SRT Parser
# =============================================================================

def parse_srt_blocks(srt_content: str) -> list:
    """Parse raw SRT content into a list of timestamp+text blocks.

    Strips sequence numbers — they are re-added after translation.
    Handles multi-line subtitle text, extra whitespace, BOM, etc.

    Returns:
        List of dicts: [{"timestamp": "00:00:01,000 --> 00:00:04,000", "text": "Hello"}, ...]
    """
    # Remove BOM if present
    srt_content = srt_content.lstrip("\ufeff")

    blocks = []
    # Split on blank lines (one or more)
    raw_blocks = re.split(r'\n\s*\n', srt_content.strip())

    for raw in raw_blocks:
        lines = raw.strip().split('\n')
        if len(lines) < 2:
            continue

        # Find the timestamp line (contains "-->")
        ts_idx = None
        for i, line in enumerate(lines):
            if '-->' in line:
                ts_idx = i
                break

        if ts_idx is None:
            continue

        timestamp = lines[ts_idx].strip()
        # Everything after the timestamp line is subtitle text
        text_lines = [l.strip() for l in lines[ts_idx + 1:] if l.strip()]
        text = '\n'.join(text_lines)

        if text:
            blocks.append({'timestamp': timestamp, 'text': text})

    return blocks


def blocks_to_srt(blocks: list, start_seq: int = 1) -> str:
    """Convert a list of {timestamp, text} blocks back into SRT format.

    Args:
        blocks:    List of dicts with 'timestamp' and 'text' keys.
        start_seq: Starting sequence number.

    Returns:
        Valid SRT string with proper sequence numbers.
    """
    lines = []
    for i, block in enumerate(blocks, start_seq):
        lines.append(str(i))
        lines.append(block['timestamp'])
        lines.append(block['text'])
        lines.append('')  # blank line separator
    return '\n'.join(lines)


def blocks_to_prompt_format(blocks: list) -> str:
    """Convert blocks to the format we send to Gemini (no sequence numbers).

    Format:
        00:00:01,000 --> 00:00:04,000
        What's good, bro?

        00:00:04,500 --> 00:00:07,000
        I'm chilling at the river bank.
    """
    parts = []
    for block in blocks:
        parts.append(f"{block['timestamp']}\n{block['text']}")
    return '\n\n'.join(parts)


# =============================================================================
# Gemini API Call (shared by both modes)
# =============================================================================

def _call_gemini(prompt: str, max_output_tokens: int = 8192, timeout: int = 120) -> str:
    """Send a prompt to Gemini 2.5 Flash and return the raw text response.

    Args:
        prompt:            The full prompt string.
        max_output_tokens: Max tokens in response (8K for chunks, 65K for full).
        timeout:           HTTP timeout in seconds.

    Returns:
        Raw text response from Gemini.

    Raises:
        ValueError: If API key is missing.
        Exception:  If Gemini API returns an error.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("❌ Error: GEMINI_API_KEY is missing from your .env file!")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "temperature": 0.3,  # faithful translation, no creative liberties
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
    }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            url, headers=headers, data=json.dumps(payload), timeout=timeout
        )
    except requests.Timeout:
        raise Exception(f"❌ Gemini API timed out after {timeout} seconds.")
    except requests.ConnectionError as e:
        raise Exception(f"❌ Connection to Gemini API failed: {e}")

    response_json = response.json()

    if "candidates" not in response_json:
        error_msg = response_json.get("error", {}).get("message", "Unknown error")
        print(f"❌ Gemini API Error: {response_json}")
        raise Exception(f"AI Translation failed: {error_msg}")

    candidate = response_json["candidates"][0]
    finish_reason = candidate.get("finishReason", "STOP")

    if finish_reason == "MAX_TOKENS":
        print("⚠️  [Gemini] WARNING: Response truncated (hit max output tokens)")
    elif finish_reason not in ("STOP", "END_TURN"):
        print(f"⚠️  [Gemini] WARNING: Unexpected finish reason: {finish_reason}")

    translated = candidate["content"]["parts"][0]["text"].strip()

    # Strip markdown code fences (common Gemini quirk)
    if translated.startswith("```"):
        lines = translated.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        translated = "\n".join(lines).strip()

    return translated


# =============================================================================
# Mode 1: Single-Shot Translation (full SRT at once)
# =============================================================================

def translate_srt(srt_content: str, target_language: str) -> str:
    """Translates an entire SRT string in one Gemini call.

    Best for short SRTs or background jobs where latency doesn't matter.
    For streaming/live use, use translate_srt_chunked() instead.
    """
    est_tokens = len(srt_content) // 4
    est_minutes = max(1, est_tokens // 10000)

    print(f"🌍 [Gemini] Translating {len(srt_content):,} chars (~{est_tokens:,} tokens) into {target_language}")
    print(f"           Estimated time: ~{est_minutes}-{est_minutes + 2} minutes")

    prompt = f"""You are an expert culturally-aware movie translator for Kalaye Subtitles. 
Translate the following movie subtitles from their original language directly into {target_language}.

CRITICAL RULES:
1. Maintain the EXACT SRT timestamp formatting. Do not change, delete, or skip any timestamps.
2. Maintain the exact sequence numbers (1, 2, 3...).
3. Use deep cultural context for slang and idioms.
4. Provide ONLY the translated SRT file in your response. No markdown, no intro, no code fences. Just the raw SRT string.

SRT CONTENT TO TRANSLATE:
{srt_content}"""

    translated = _call_gemini(prompt, max_output_tokens=65536, timeout=600)
    print(f"✅ [Gemini] Translation complete! {len(translated):,} chars returned")
    return translated


# =============================================================================
# Mode 2: Streaming Chunked Translation (yields chunks as they're ready)
# =============================================================================

CHUNK_SIZE = 50  # subtitle blocks per chunk (~3-5 min of screen time)
CONTEXT_LINES = 3  # number of previously-translated lines to include as context


def translate_srt_chunked(srt_content: str, target_language: str, chunk_size: int = CHUNK_SIZE):
    """Generator that yields translated SRT chunks as Gemini completes them.

    This is the core of the streaming translation system. Each chunk:
      1. Contains ~50 subtitle blocks (~3-5 min of movie)
      2. Includes the last 3 translated lines from the previous chunk as
         narrative context (so Gemini maintains coherence across chunks)
      3. Gets sequence numbers stripped before sending and re-added after

    Yields:
        dict with keys:
            chunk_index      — 0, 1, 2, ...
            total_chunks     — total number of chunks
            srt_chunk        — translated SRT string (with correct sequence numbers)
            blocks_in        — number of source blocks sent
            blocks_out       — number of translated blocks received
            is_final         — True if this is the last chunk
            elapsed_seconds  — how long this chunk took to translate
    """
    # Parse the full SRT into blocks
    all_blocks = parse_srt_blocks(srt_content)
    total_blocks = len(all_blocks)

    if total_blocks == 0:
        print("⚠️  [Translator] No subtitle blocks found in input SRT")
        return

    total_chunks = (total_blocks + chunk_size - 1) // chunk_size

    print(f"🌍 [Gemini] Streaming translation: {total_blocks} blocks → "
          f"{total_chunks} chunks of {chunk_size} into {target_language}")

    context_texts = []  # last N translated text lines for next chunk's context
    global_seq = 1      # running sequence number across all chunks

    for chunk_idx in range(total_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, total_blocks)
        chunk_blocks = all_blocks[start:end]

        # Build the subtitle content (no sequence numbers)
        chunk_content = blocks_to_prompt_format(chunk_blocks)

        # Build context from previous chunk
        context_section = ""
        if context_texts:
            context_section = (
                "\n\nCONTEXT FROM PREVIOUS SCENE (for narrative continuity — "
                "DO NOT include these lines in your output, they are already translated):\n"
            )
            for ct in context_texts:
                context_section += f'  "{ct}"\n'

        # Build prompt
        prompt = f"""You are an expert culturally-aware movie translator for Kalaye Subtitles.
Translate the following movie subtitle chunk from its original language directly into {target_language}.

CRITICAL RULES:
1. TIMESTAMPS MUST USE COMMAS FOR MILLISECONDS. Format: HH:MM:SS,mmm --> HH:MM:SS,mmm
   Correct: 00:01:03,600 --> 00:01:05,040
   Wrong: 01:03:600 --> 01:05:040
2. Do NOT change, skip, or merge any timestamps. Every input block MUST appear in the output.
3. Do NOT include sequence numbers. Output ONLY timestamp lines followed by translated text.
4. Use deep cultural context for slang, idioms, and colloquial speech.
5. Output ONLY the translated subtitle blocks. No markdown, no intro, no explanation.
{context_section}
SUBTITLE CHUNK TO TRANSLATE:
{chunk_content}"""

        # Call Gemini
        t0 = time.time()
        print(f"  📤 Chunk {chunk_idx + 1}/{total_chunks} "
              f"({len(chunk_blocks)} blocks, chars={len(chunk_content)})...", flush=True)

        try:
            raw_response = _call_gemini(prompt, max_output_tokens=8192, timeout=120)
        except Exception as e:
            print(f"  ❌ Chunk {chunk_idx + 1} failed: {e}")
            # Yield an error event so the client knows
            yield {
                "chunk_index": chunk_idx,
                "total_chunks": total_chunks,
                "srt_chunk": "",
                "blocks_in": len(chunk_blocks),
                "blocks_out": 0,
                "is_final": chunk_idx == total_chunks - 1,
                "elapsed_seconds": round(time.time() - t0, 1),
                "error": str(e),
            }
            continue

        elapsed = time.time() - t0

        # Parse Gemini's response back into blocks
        translated_blocks = parse_srt_blocks(raw_response)

        # 🛡️ SAFETY GUARD: If Gemini returned the exact same number of blocks,
        # forcefully overwrite its timestamps with the original perfect timestamps
        # to guarantee it NEVER corrupts them.
        if len(translated_blocks) == len(chunk_blocks):
            for i in range(len(translated_blocks)):
                translated_blocks[i]["timestamp"] = chunk_blocks[i]["timestamp"]
        else:
            print(f"  ⚠️ Block count mismatch: Sent {len(chunk_blocks)}, got {len(translated_blocks)}")

        # Re-number with global sequence numbers
        result_srt = blocks_to_srt(translated_blocks, start_seq=global_seq)
        global_seq += len(translated_blocks)

        # Update context for next chunk (last N translated texts)
        if translated_blocks:
            context_texts = [b["text"] for b in translated_blocks[-CONTEXT_LINES:]]

        print(f"  ✅ Chunk {chunk_idx + 1}/{total_chunks} done — "
              f"{len(translated_blocks)} blocks in {elapsed:.1f}s", flush=True)

        yield {
            "chunk_index": chunk_idx,
            "total_chunks": total_chunks,
            "srt_chunk": result_srt,
            "blocks_in": len(chunk_blocks),
            "blocks_out": len(translated_blocks),
            "is_final": chunk_idx == total_chunks - 1,
            "elapsed_seconds": round(elapsed, 1),
        }

    print(f"🏁 [Gemini] Streaming translation complete — {global_seq - 1} total blocks")


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    sample_srt = """1
00:00:01,000 --> 00:00:04,000
What's good, bro?

2
00:00:04,500 --> 00:00:07,000
I'm chilling at the river bank.

3
00:00:08,000 --> 00:00:11,000
You know what I mean? This place is fire.
"""

    user_selected_language = "Yoruba"

    print("=== Single-shot mode ===")
    try:
        result = translate_srt(sample_srt, user_selected_language)
        print(f"\n✅ Result:\n{result}")
    except Exception as e:
        print(e)

    print("\n=== Chunked mode (chunk_size=2 for testing) ===")
    try:
        full_srt = ""
        for chunk in translate_srt_chunked(sample_srt, user_selected_language, chunk_size=2):
            print(f"  Received chunk {chunk['chunk_index']}: {chunk['blocks_out']} blocks")
            full_srt += chunk["srt_chunk"] + "\n\n"
        print(f"\n✅ Full stitched result:\n{full_srt}")
    except Exception as e:
        print(e)
