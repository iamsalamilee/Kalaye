"""
test_providers.py — Test each LLM provider individually.

Run: python test_providers.py
"""

import os
import sys
import time
from dotenv import load_dotenv
load_dotenv()

# Add parent dir so we can import translator helpers
sys.path.insert(0, os.path.dirname(__file__))
from server.translator import (
    _call_gemini_raw,
    _call_groq_raw,
    _call_cerebras_raw,
    _call_deepseek_raw,
)

# ─── Simple test prompt ───────────────────────────────────────────────────────
PROMPT = """Translate the following single subtitle block from English into Yoruba.
Output ONLY the translated text. No explanations.

"What's good, bro? I'm chilling at the river bank." """

# ─── Providers to test ────────────────────────────────────────────────────────
PROVIDERS = [
    ("Gemini 2.5 Flash",  "GEMINI_API_KEY",   _call_gemini_raw),
    ("Groq Llama3-70b",   "GROQ_API_KEY",     _call_groq_raw),
    ("Cerebras Llama3.1", "CEREBRAS_API_KEY",  _call_cerebras_raw),
    ("DeepSeek Chat",     "DEEPSEEK_API_KEY",  _call_deepseek_raw),
]

# ─── Run tests ─────────────────────────────────────────────────────────────
def run_tests():
    print("=" * 60)
    print("  Kalaye — LLM Provider Test")
    print("=" * 60)

    results = []
    for name, env_key, fn in PROVIDERS:
        key_present = bool(os.environ.get(env_key))
        print(f"\n🔌 Testing: {name}")
        print(f"   API Key ({env_key}): {'✅ found' if key_present else '❌ MISSING — skipping'}")

        if not key_present:
            results.append((name, "SKIPPED", "No API key", 0))
            continue

        try:
            t0 = time.time()
            response = fn(PROMPT, max_output_tokens=256, timeout=30)
            elapsed = time.time() - t0
            print(f"   ✅ Response ({elapsed:.1f}s): {response[:120]}...")
            results.append((name, "OK", response[:60], elapsed))
        except Exception as e:
            elapsed = time.time() - t0
            print(f"   ❌ Failed ({elapsed:.1f}s): {e}")
            results.append((name, "FAILED", str(e)[:80], elapsed))

    # ─── Summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Test Summary")
    print("=" * 60)
    print(f"  {'Provider':<22} {'Status':<10} {'Time':>6}")
    print(f"  {'-'*22} {'-'*10} {'-'*6}")
    for name, status, _, elapsed in results:
        icon = "✅" if status == "OK" else ("⏭️ " if status == "SKIPPED" else "❌")
        time_str = f"{elapsed:.1f}s" if elapsed else "  —"
        print(f"  {icon} {name:<20} {status:<10} {time_str:>6}")

    ok = sum(1 for _, s, _, _ in results if s == "OK")
    total = sum(1 for _, s, _, _ in results if s != "SKIPPED")
    print(f"\n  {ok}/{total} providers working\n")


if __name__ == "__main__":
    run_tests()
