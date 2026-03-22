# Ghost Sync — Daily Timeline (1 Hour/Day)

> **Started:** March 21, 2026
> **Current Phase:** Phase 2 — Desktop MVP
> **Pace:** ~1 hour per day

---

## ✅ Phase 1 — Prove The Math (Complete)

| Day | Date | Task | Status |
|-----|------|------|--------|
| 1 | Mar 21 (Fri) | Set up Python venv, install deps, install ffmpeg | ✅ |
| 2 | Mar 22 (Sat) | First sync test passed (0.05s error), push to GitHub | ✅ |

> Phase 1 validated: FFT cross-correlation works. Additional testing will happen alongside development.

---

## Phase 2 — Desktop MVP (Mar 23 → ~May 4)

### Week 1: Server Backend (Mar 23 – Mar 29)

| Day | Date | Task | Status |
|-----|------|------|--------|
| 3 | Mar 23 (Sun) | FastAPI project setup, folder structure, basic endpoints | ⬜ |
| 4 | Mar 24 (Mon) | Database schema (Postgres/SQLite), movie + subtitle tables | ⬜ |
| 5 | Mar 25 (Tue) | Filename parser — extract movie name/year from filenames | ⬜ |
| 6 | Mar 26 (Wed) | OpenSubtitles API — fetch SRT candidates | ⬜ |
| 7 | Mar 27 (Thu) | Sync endpoint — run FFT algo server-side on uploaded audio | ⬜ |
| 8 | Mar 28 (Fri) | Multi-SRT selection — pick best of top 3 candidates | ⬜ |
| 9 | Mar 29 (Sat) | Server integration test — full pipeline end-to-end | ⬜ |

### Week 2: Desktop Client Basics (Mar 30 – Apr 5)

| Day | Date | Task | Status |
|-----|------|------|--------|
| 10 | Mar 30 (Sun) | PyQt5 setup, basic floating overlay window | ⬜ |
| 11 | Mar 31 (Mon) | Click-through, always-on-top overlay | ⬜ |
| 12 | Apr 1 (Tue) | Subtitle text rendering + styling | ⬜ |
| 13 | Apr 2 (Wed) | Video player detection with psutil | ⬜ |
| 14 | Apr 3 (Thu) | Audio extraction from local video files | ⬜ |
| 15 | Apr 4 (Fri) | WASAPI loopback capture (streaming mode) | ⬜ |
| 16 | Apr 5 (Sat) | Connect client → server API calls | ⬜ |

### Week 3: Integration + Pre-Check Ladder (Apr 6 – Apr 12)

| Day | Date | Task | Status |
|-----|------|------|--------|
| 17 | Apr 6 (Sun) | Local SQLite cache (check before calling server) | ⬜ |
| 18 | Apr 7 (Mon) | Pre-check ladder: chain cache → hash → filename → fingerprint | ⬜ |
| 19 | Apr 8 (Tue) | ACRCloud integration (audio fingerprinting) | ⬜ |
| 20 | Apr 9 (Wed) | Whisper API fallback (last resort identification) | ⬜ |
| 21 | Apr 10 (Thu) | Audio compression: Opus 16kbps mono pipeline | ⬜ |
| 22 | Apr 11 (Fri) | End-to-end: open video → subtitles appear | ⬜ |
| 23 | Apr 12 (Sat) | Bug fixes + error handling | ⬜ |

### Week 4: Polish + Deploy (Apr 13 – Apr 19)

| Day | Date | Task | Status |
|-----|------|------|--------|
| 24 | Apr 13 (Sun) | Overlay polish — font, size, position, transparency | ⬜ |
| 25 | Apr 14 (Mon) | Settings UI — language, overlay position, delay adjust | ⬜ |
| 26 | Apr 15 (Tue) | Deploy server to VPS (DigitalOcean/Railway) | ⬜ |
| 27 | Apr 16 (Wed) | Test 5 movies in a row (pass criteria) | ⬜ |
| 28 | Apr 17 (Thu) | Fix failures from 5-movie test | ⬜ |
| 29 | Apr 18 (Fri) | Record internal demo video | ⬜ |
| 30 | Apr 19 (Sat) | Phase 2 review, push everything to GitHub | ⬜ |

---

## Progress Tracker

| Phase | Target End | Status |
|-------|-----------|--------|
| **Phase 1** — Prove the Math | Mar 22 | ✅ Complete |
| **Phase 2** — Desktop MVP | Apr 19 | ⬜ Starting |
| **Phase 3** — Translation + Community | Jun–Jul | ⬜ |
| **Phase 4** — Android Client | Aug–Oct | ⬜ |
| **Phase 5** — Scale + Launch | Nov–Feb 2027 | ⬜ |

---

*Updated: March 22, 2026*
