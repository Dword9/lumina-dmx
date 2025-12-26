# AI_RULES.md — Lumina Backend (realtime audio + DMX/ArtNet)

This repository contains a realtime audio + DSP + DMX/ArtNet backend.
Audio stability is the top priority. Any AI-assisted edits MUST respect the invariants below.

## 0. Prime Directive (non-negotiable)

**Audio is absolutely stable.**
Nothing (UI, networking, WebSocket, analytics, ML) is allowed to block, stall, or alter audio timing.

If a change risks jitter/underruns: **do not do it**.

---

## 1. Do-Not-Touch List (realtime critical)

AI MUST NOT modify these without explicit human approval and a dedicated test plan:

- Audio callback functions (sounddevice callbacks)
- AudioEngine realtime scheduling / pacing logic
- File playback pacing / prebuffer logic (underrun must become silence, never crackle)
- DSP core hot path (FFT / meter computations)
- ArtNet/DMX send loop timing (~40 FPS) and non-blocking UDP behavior
- Threading / queue mechanics on the audio path
- Any code executed on every audio block / callback tick

If you must propose a change here, only provide an **analysis + minimal diff plan**, not a rewrite.

---

## 2. Allowed Edit Zones (generally safe)

AI may modify these with care (still no breaking changes):

- UI/WS telemetry formatting (adding new message `type` only)
- Non-realtime utilities, logging, documentation
- WebSocket client/server routing (but keep protocol stable)
- Config parsing and feature flags (default must preserve current behavior)
- ML worker/inference code **ONLY if it is strictly side-chain** (see §4)

---

## 3. Protocol Compatibility Rules (WebSocket / HTTP)

### WebSocket (fixed contract)
- No breaking changes.
- Only allowed extension: **add new message `type`**.
- Existing `type` payload schemas must remain backward compatible.
- If `reqId` is received, it must be echoed in responses.

### HTTP upload
- `/upload` behavior must remain compatible (dedup by sha256, extract audio from video if present, registry entry, return `sourceId`).

---

## 4. ML Integration Rules (side-chain only)

ML MUST NOT affect audio.

### Required architecture
- Inference runs in a separate thread/task.
- Communication from audio/DSP → ML uses a **bounded queue**.
- Enqueue MUST be non-blocking:
  - `put_nowait` / try-enqueue only
  - drop on overload (never wait)
- ML failure or overload must not impact audio or DSP.
- ML publishes telemetry events only:
  - `type: "ai_classifier_event"` (or new telemetry types if added)
  - payload must not include raw audio

### Forbidden
- Any waiting/blocking in audio callback.
- Any backpressure from ML into audio.
- Any synchronous inference on the DSP hot path.

---

## 5. Performance & Safety Rules

### No new hot loops
- Do not add new loops/timers that run at audio rate or high frequency in Python main thread.
- Avoid excessive logging in hot paths.

### No allocations in callbacks
- Audio callbacks must avoid heavy allocations, I/O, locks, JSON, logging.

### Concurrency
- Do not introduce locks that can be contended by audio callbacks.
- Prefer lock-free / minimal-lock patterns for ring buffers and queues.

---

## 6. Editing Workflow Rules (for AI tools)

Because some AI tools can only rewrite full files:

1. **Preserve public interfaces**
   - Do not rename existing functions/classes used elsewhere.
   - Do not remove methods called by `control.py` or server routing.
2. **Keep defaults identical**
   - If a feature is off by default, it must stay off by default.
3. **One change per commit**
   - Small, reversible edits.
4. **No “cleanup refactors”**
   - Do not reformat, reorder, or “modernize” unrelated code.
5. **Add comments for invariants**
   - If touching sensitive areas, add a short comment explaining why it’s safe.

---

## 7. Required Tests Before Merging Realtime Changes

For any change that touches audio/DSP/WS/ArtNet:

- Run file playback for 3–5 minutes
- Run input device capture for 3–5 minutes (if available)
- Keep WS connected and telemetry enabled
- Confirm:
  - no audible glitches
  - no underrun spikes
  - CPU stable (no runaway thread creation)
  - WS message rate not exploding

If tests cannot be run, do not merge changes to `main`.

---

## 8. If in doubt

If you are not 100% sure a change is realtime-safe:
- Do not implement it.
- Provide a minimal plan and ask for explicit approval.

**Audio stability > code style > features.**
