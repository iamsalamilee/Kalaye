"""
Kalaye — Subtitle Translator (v3: Health-Aware Multi-Provider)
==============================================================
Translates SRT subtitles using a smart provider waterfall.

Provider Order:
  FREE TIER  (tried first, in order):
    1. Gemini 2.5 Flash   — best quality
    2. Groq Llama3.3-70b  — fastest
    3. Cerebras Qwen3-235b — Groq alternative

  PAID TIER  (auto-activated when all free providers are exhausted):
    4. Amazon Nova Pro    — $1000 credit, high quality
    5. DeepSeek Chat      — ultra cheap (~$0.01/movie)
    6. Together AI        — reliable paid fallback

Failure Handling:
  • HTTP 429 (quota/rate limit hit) → provider blacklisted for 24 hours
  • HTTP 503 (overloaded/busy)      → provider cooled down for 5 minutes
  • Any other error                 → provider cooled down for 5 minutes
  • All providers exhausted         → raises a clear final error

Two translation modes:
  1. translate_srt()         — Single-shot, full SRT in one call.
  2. translate_srt_chunked() — Generator, streams ~50-block chunks (used by SSE).
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
# Provider Health Tracker
# =============================================================================

# Cooldown durations
_COOLDOWN_SHORT  = 5 * 60       # 5 minutes  — for 503 / transient errors
_COOLDOWN_LONG   = 24 * 60 * 60 # 24 hours   — for 429 / daily quota exhausted

# { provider_name: {"until": float_timestamp, "reason": str} }
_provider_health: dict = {}


def _is_healthy(provider: str) -> bool:
    """Return True if the provider is not in a cooldown period."""
    if provider not in _provider_health:
        return True
    return time.time() >= _provider_health[provider]["until"]


def _mark_sick(provider: str, duration: float, reason: str):
    """Put a provider into cooldown."""
    until = time.time() + duration
    _provider_health[provider] = {"until": until, "reason": reason}
    friendly = f"{int(duration // 60)}m" if duration < 3600 else "24h"
    print(f"  🚫 [{provider}] marked unavailable for {friendly}: {reason}")


def _mark_healthy(provider: str):
    """Clear a provider's cooldown (called on success)."""
    if provider in _provider_health:
        del _provider_health[provider]
        print(f"  ✅ [{provider}] is healthy again")


def _get_provider_status() -> dict:
    """Return a snapshot of all provider health states (for logging)."""
    now = time.time()
    status = {}
    for name, info in _provider_health.items():
        remaining = max(0, info["until"] - now)
        status[name] = f"sick ({int(remaining)}s remaining) — {info['reason']}"
    return status


# =============================================================================
# Shared Response Cleaner
# =============================================================================

def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences that some LLMs add around their output."""
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text



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
# Individual Provider API Calls
# =============================================================================

def _call_gemini_raw(prompt: str, max_output_tokens: int = 8192, timeout: int = 120) -> str:
    """Call Gemini 2.5 Flash. Raises with status_code attribute on HTTP errors."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from your .env file!")

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_output_tokens, "temperature": 0.3},
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
    }
    try:
        r = requests.post(url, headers={"Content-Type": "application/json"},
                          data=json.dumps(payload), timeout=timeout)
    except requests.Timeout:
        raise Exception(f"Gemini timed out after {timeout}s.")
    except requests.ConnectionError as e:
        raise Exception(f"Connection to Gemini failed: {e}")

    rj = r.json()
    if "candidates" not in rj:
        err = rj.get("error", {})
        ex = Exception(err.get("message", "Unknown Gemini error"))
        ex.status_code = err.get("code", r.status_code)
        raise ex

    candidate = rj["candidates"][0]
    if candidate.get("finishReason") == "MAX_TOKENS":
        print("⚠️  [Gemini] Response truncated — hit max output tokens")

    return _strip_code_fences(candidate["content"]["parts"][0]["text"].strip())


def _call_groq_raw(prompt: str, max_output_tokens: int = 8192, timeout: int = 120) -> str:
    """Call Groq Llama-3.3-70b. Raises with status_code attribute on HTTP errors."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from your .env file!")

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "llama-3.3-70b-versatile",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.3,
                  "max_tokens": min(max_output_tokens, 8192)},
            timeout=timeout,
        )
    except requests.Timeout:
        raise Exception(f"Groq timed out after {timeout}s.")
    except requests.ConnectionError as e:
        raise Exception(f"Connection to Groq failed: {e}")

    rj = r.json()
    if "error" in rj:
        ex = Exception(rj["error"].get("message", "Unknown Groq error"))
        ex.status_code = r.status_code
        raise ex

    if "choices" not in rj:
        raise Exception(f"Groq unexpected response: {str(rj)[:200]}")

    return _strip_code_fences(rj["choices"][0]["message"]["content"].strip())


def _call_cerebras_raw(prompt: str, max_output_tokens: int = 8192, timeout: int = 120) -> str:
    """Call Cerebras Llama-3.3-70b (OpenAI-compatible). Raises with status_code on errors."""
    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        raise ValueError("CEREBRAS_API_KEY is missing from your .env file!")

    try:
        r = requests.post(
            "https://api.cerebras.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "qwen-3-235b-a22b-instruct-2507",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.3,
                  "max_tokens": min(max_output_tokens, 8192)},
            timeout=timeout,
        )
    except requests.Timeout:
        raise Exception(f"Cerebras timed out after {timeout}s.")
    except requests.ConnectionError as e:
        raise Exception(f"Connection to Cerebras failed: {e}")

    rj = r.json()
    if "error" in rj:
        ex = Exception(rj["error"].get("message", "Unknown Cerebras error"))
        ex.status_code = r.status_code
        raise ex

    if "choices" not in rj:
        raise Exception(f"Cerebras unexpected response: {str(rj)[:200]}")

    return _strip_code_fences(rj["choices"][0]["message"]["content"].strip())


def _call_deepseek_raw(prompt: str, max_output_tokens: int = 8192, timeout: int = 120) -> str:
    """Call DeepSeek Chat (paid, ~$0.01/movie). Raises with status_code on errors."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is missing from your .env file!")

    try:
        r = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "deepseek-chat",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.3,
                  "max_tokens": min(max_output_tokens, 8192)},
            timeout=timeout,
        )
    except requests.Timeout:
        raise Exception(f"DeepSeek timed out after {timeout}s.")
    except requests.ConnectionError as e:
        raise Exception(f"Connection to DeepSeek failed: {e}")

    rj = r.json()
    if "error" in rj:
        ex = Exception(rj["error"].get("message", "Unknown DeepSeek error"))
        ex.status_code = r.status_code
        raise ex

    return _strip_code_fences(rj["choices"][0]["message"]["content"].strip())


def _call_together_raw(prompt: str, max_output_tokens: int = 8192, timeout: int = 120) -> str:
    """Call Together AI Llama-3.3-70B (paid, ~$0.18/1M tokens). Raises with status_code on errors."""
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY is missing from your .env file!")

    try:
        r = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.3,
                  "max_tokens": min(max_output_tokens, 8192)},
            timeout=timeout,
        )
    except requests.Timeout:
        raise Exception(f"Together AI timed out after {timeout}s.")
    except requests.ConnectionError as e:
        raise Exception(f"Connection to Together AI failed: {e}")

    rj = r.json()
    if "error" in rj:
        ex = Exception(rj["error"].get("message", "Unknown Together AI error"))
        ex.status_code = r.status_code
        raise ex

    return _strip_code_fences(rj["choices"][0]["message"]["content"].strip())


def _call_nova_raw(prompt: str, max_output_tokens: int = 8192, timeout: int = 120) -> str:
    """Call Amazon Nova Pro via AWS Bedrock. Uses boto3 (AWS SDK)."""
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    region = os.environ.get("AWS_REGION", "us-east-1")

    if not access_key or not secret_key:
        raise ValueError("AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY missing from .env!")

    try:
        import boto3
    except ImportError:
        raise ValueError("boto3 not installed — run: pip install boto3")

    try:
        client = boto3.client(
            'bedrock-runtime',
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

        body = json.dumps({
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {
                "maxTokens": max_output_tokens,
                "temperature": 0.3,
            }
        })

        response = client.invoke_model(
            modelId='amazon.nova-pro-v1:0',
            body=body,
            contentType='application/json',
            accept='application/json',
        )

        result = json.loads(response['body'].read())
        return _strip_code_fences(result['output']['message']['content'][0]['text'].strip())

    except Exception as e:
        # Extract HTTP status code if available
        status_code = getattr(e, 'response', {}).get('ResponseMetadata', {}).get('HTTPStatusCode', None)
        if status_code:
            ex = Exception(str(e))
            ex.status_code = status_code
            raise ex
        raise Exception(f"Nova Bedrock error: {e}")


# =============================================================================
# Smart Health-Aware Provider Router
# =============================================================================

# Ordered provider list: (name, tier, callable)
# "free"  → try first; 429 = 24h blacklist, 503 = 5min cooldown
# "paid"  → auto-activated only when ALL free providers are unavailable
_PROVIDERS = [
    ("Nova",     "free", _call_nova_raw),
    ("Gemini",   "free", _call_gemini_raw),
    ("Groq",     "free", _call_groq_raw),
    ("Cerebras", "free", _call_cerebras_raw),
    ("DeepSeek", "paid", _call_deepseek_raw),
]


def _call_llm(prompt: str, max_output_tokens: int = 8192, timeout: int = 120) -> str:
    """
    Health-aware multi-provider LLM router.

    Routing logic:
      1. Try all FREE providers in order (skip any that are in cooldown).
      2. If ALL free providers are unavailable, automatically escalate to PAID providers.
      3. If ALL providers fail, raise a descriptive final error.

    Cooldown rules (applied per provider):
      • HTTP 429 (daily quota / rate limit) → 24-hour blacklist
      • HTTP 503 / any transient error      → 5-minute cooldown
    """
    errors = []
    free_providers  = [(n, fn) for n, t, fn in _PROVIDERS if t == "free"]
    paid_providers  = [(n, fn) for n, t, fn in _PROVIDERS if t == "paid"]

    def _try_providers(provider_list: list, label: str) -> str | None:
        for name, fn in provider_list:
            if not _is_healthy(name):
                print(f"  [skip] [{name}] skipped — in cooldown")
                continue
            print(f"  [try] [{name}] attempting translation...")
            
            # Nova gets 5 retries, others get 3. All 10s apart.
            max_retries = 5 if name == "Nova" else 3
            retry_delay = 10
            
            for attempt in range(max_retries):
                try:
                    result = fn(prompt, max_output_tokens, timeout)
                    _mark_healthy(name)
                    print(f"  [ok] [{name}] success")
                    return result
                except ValueError as e:
                    print(f"  [warn] [{name}] skipped — {e}")
                    errors.append(f"{name}: {e}")
                    break  # no point retrying a missing key
                except Exception as e:
                    status_code = getattr(e, "status_code", None)
                    err_msg = str(e).lower()
                    
                    is_network_error = any(kw in err_msg for kw in [
                        "connection", "nameresolution", "timeout", 
                        "could not connect", "max retries", "ssl"
                    ])
                    
                    if is_network_error and attempt < max_retries - 1:
                        print(f"  [retry] [{name}] network error, retrying in {retry_delay}s... (attempt {attempt + 2}/{max_retries})")
                        time.sleep(retry_delay)
                        continue
                    
                    # All retries exhausted or non-network error
                    if status_code == 429:
                        if "daily" in err_msg or "quota" in err_msg:
                            _mark_sick(name, _COOLDOWN_LONG, f"429 daily quota exhausted")
                        else:
                            _mark_sick(name, 60, f"429 rate limited (short cooldown)")
                    elif status_code == 400:
                        _mark_sick(name, _COOLDOWN_SHORT, f"400: {e}")
                    elif is_network_error:
                        _mark_sick(name, 30, f"network error: {e}")
                    else:
                        _mark_sick(name, _COOLDOWN_SHORT, f"{status_code or 'error'}: {e}")
                    errors.append(f"{name}: {e}")
        return None

    # --- Stage 1: Free tier ---
    # Tries Gemini → Groq → Cerebras in order.
    # Each is only tried if it is not currently in cooldown.
    # Returns immediately on first success, skips on 429/503.
    result = _try_providers(free_providers, "FREE")
    if result is not None:
        return result

    # --- Stage 2: Paid tier (auto-escalate) ---
    # This is only reached if ALL free providers either:
    #   a) failed this request (5-min cooldown), or
    #   b) hit their daily quota (24-hr blacklist)
    # FIX: was `n, _, _` which bound `_` to the function ref, making `_ == "free"` always False
    all_free_sick = all(not _is_healthy(n) for n, t, fn in _PROVIDERS if t == "free")
    any_free_healthy = any(_is_healthy(n) for n, t, fn in _PROVIDERS if t == "free")

    if all_free_sick:
        print("  💳 All 3 free providers are blacklisted — escalating to PAID tier (DeepSeek)...")
    elif not any_free_healthy:
        print("  💳 No free provider could serve this request — escalating to PAID tier...")
    
    # Confirm which free providers were tried before reaching this point
    for n, t, _ in _PROVIDERS:
        if t == "free" and n in _provider_health:
            info = _provider_health[n]
            remaining = max(0, info["until"] - time.time())
            label = "24h quota" if remaining > 3600 else f"{int(remaining//60)}min cooldown"
            print(f"  ℹ️  [{n}] confirmed unavailable: {label}")

    result = _try_providers(paid_providers, "PAID")
    if result is not None:
        return result

    # --- Stage 3: Total failure — every single provider is exhausted ---
    status_snapshot = _get_provider_status()
    status_lines = "\n".join(f"  • {k}: {v}" for k, v in status_snapshot.items()) or "  (no status recorded)"
    raise Exception(
        f"❌ Translation failed — all providers exhausted.\n\n"
        f"Provider status:\n{status_lines}\n\n"
        f"Individual errors:\n" + "\n".join(f"  • {e}" for e in errors)
    )


# =============================================================================
# Mode 1: Single-Shot Translation (full SRT at once)
# =============================================================================

def translate_srt(srt_content: str, target_language: str) -> str:
    """Translates an entire SRT string using the smart provider router.

    Best for short SRTs or background jobs where latency doesn't matter.
    For streaming/live use, use translate_srt_chunked() instead.
    """
    est_tokens = len(srt_content) // 4
    est_minutes = max(1, est_tokens // 10000)
    print(f"🌍 [Translate] Translating {len(srt_content):,} chars (~{est_tokens:,} tokens) into {target_language}")
    print(f"              Estimated time: ~{est_minutes}-{est_minutes + 2} minutes")

    prompt = f"""You are an expert culturally-aware movie translator for Kalaye Subtitles.
Translate the following movie subtitles from their original language directly into {target_language}.

CRITICAL RULES:
1. Maintain the EXACT SRT timestamp formatting. Do not change, delete, or skip any timestamps.
2. Maintain the exact sequence numbers (1, 2, 3...).
3. Use deep cultural context for slang and idioms.
4. Provide ONLY the translated SRT file in your response. No markdown, no intro, no code fences. Just the raw SRT string.

SRT CONTENT TO TRANSLATE:
{srt_content}"""

    translated = _call_llm(prompt, max_output_tokens=65536, timeout=600)
    print(f"✅ [Translate] Complete! {len(translated):,} chars returned")
    return translated



# =============================================================================
# Mode 2: Streaming Chunked Translation (yields chunks as they're ready)
# =============================================================================

CHUNK_SIZE = 50    # subtitle blocks per chunk (~3-5 min of screen time)
CONTEXT_LINES = 3  # number of previously-translated lines to include as context


def translate_srt_chunked(srt_content: str, target_language: str, chunk_size: int = CHUNK_SIZE):
    """Generator that yields translated SRT chunks as they complete.

    Uses the health-aware multi-provider router for each chunk, so if one
    provider fails mid-movie it silently falls to the next healthy one.

    Each chunk:
      1. Contains ~50 subtitle blocks (~3-5 min of movie)
      2. Includes the last 3 translated lines from the previous chunk as
         narrative context so the LLM maintains coherence across chunks
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

    print(f"🌍 [Translate] Streaming: {total_blocks} blocks → "
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

        t0 = time.time()
        print(f"  📤 Chunk {chunk_idx + 1}/{total_chunks} "
              f"({len(chunk_blocks)} blocks, chars={len(chunk_content)})...", flush=True)

        try:
            raw_response = _call_llm(prompt, max_output_tokens=8192, timeout=120)
        except Exception as e:
            print(f"  ❌ Chunk {chunk_idx + 1} failed (all providers exhausted): {e}")
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

        # Parse the response back into blocks
        translated_blocks = parse_srt_blocks(raw_response)

        # SAFETY GUARD: Force-overwrite timestamps with the originals.
        # This guarantees timestamps are NEVER corrupted by the LLM.
        if len(translated_blocks) == len(chunk_blocks):
            # Perfect match — overwrite all timestamps
            for i in range(len(translated_blocks)):
                translated_blocks[i]["timestamp"] = chunk_blocks[i]["timestamp"]
        elif len(translated_blocks) > 0:
            # Block count mismatch — LLM merged/split/skipped some blocks.
            # Force-apply original timestamps to however many we can map.
            print(f"  Warning: Block count mismatch: Sent {len(chunk_blocks)}, got {len(translated_blocks)}")
            min_len = min(len(translated_blocks), len(chunk_blocks))
            for i in range(min_len):
                translated_blocks[i]["timestamp"] = chunk_blocks[i]["timestamp"]
            # If LLM returned MORE blocks than expected, use the last original timestamp
            for i in range(min_len, len(translated_blocks)):
                translated_blocks[i]["timestamp"] = chunk_blocks[-1]["timestamp"]

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

    print(f"🏁 [Translate] Streaming complete — {global_seq - 1} total blocks")


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
