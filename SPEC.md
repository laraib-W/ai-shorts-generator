# AI Text-to-Video Generator — Project Specification

**Version:** 0.1 (MVP)
**Last updated:** June 2026
**Base project:** Fork of [MoneyPrinterTurbo](https://github.com/harry0703/MoneyPrinterTurbo) (Python, MIT License)

---

## 1. Overview

A web service that converts a user's raw idea ("vision") into a fully generated short-form video (MP4) suitable for YouTube Shorts, TikTok, and Instagram Reels. The system refines the vision into a script, splits it into scenes, generates visuals and voiceover per scene, assembles everything into a video, and returns a downloadable MP4.

Users bring their own API keys for AI services, so the platform's operating cost stays near zero. Users who supply no image-generation key fall back to free stock footage.

### Goals
- Turn a short text vision into a finished, narrated, captioned MP4 with minimal user input.
- Keep platform infrastructure cost low (~$5–10/month).
- Pass all AI generation cost to the user via their own API keys.
- Keep the user experience simple — no model jargon, sensible defaults, works out of the box.

---

## 2. User story

> As a user, I want to submit a vision and receive a fully generated short video, so that I can create YouTube Shorts / TikTok content automatically.

---

## 3. Pipeline

The pipeline is a sequence of stages orchestrated by **LangGraph** (see Section 4). MoneyPrinterTurbo already implements most of the underlying stage logic; new work is concentrated in the image-generation stage and the graph that wires everything together with retries and fallback.

| # | Stage | Description | Source |
|---|-------|-------------|--------|
| 1 | Input | Accept raw vision + user API keys + settings via API | New (thin layer) |
| 2 | Script refinement | LLM improves clarity, pacing, engagement for short-form video | MPT (existing) |
| 3 | Scene segmentation | LLM splits script into scenes with narration + duration hints | MPT (existing) |
| 4 | Image prompt generation | LLM writes a cinematic image prompt per scene | New / extend |
| 5 | Visual generation | AI image per scene (user key) OR Pexels stock fallback | **New — primary build** |
| 6 | Voiceover | TTS from final narration | MPT (existing) |
| 7 | Assembly | FFmpeg + MoviePy stitch visuals + audio + subtitles + music | MPT (existing) |
| 8 | Output | Upload MP4, return downloadable URL | New (storage layer) |

---

## 4. Orchestration — LangGraph

The pipeline is orchestrated with **LangGraph**. Rather than wrapping MoneyPrinterTurbo's linear flow as-is, the graph uses LangGraph's stateful, branching, and looping capabilities — which is what justifies the dependency. MoneyPrinterTurbo's individual stage functions (script gen, TTS, FFmpeg assembly, Pexels fetch) are wrapped as callable units invoked inside graph nodes.

### 4.1 Why LangGraph (capabilities used)
- **Per-scene retry loop** — if image generation fails, or a vision check finds the image doesn't match its prompt, loop back and regenerate (bounded by a max-retry cap) before falling back to Pexels.
- **Script-critic loop** — generate script → critique → if quality is below threshold, rewrite. Bounded iterations.
- **Conditional fallback routing** — the AI-vs-stock decision is a graph edge, not an `if` buried in a function.
- **Shared typed state** — vision, refined script, scenes, per-scene assets, settings, and keys live in one state object passed between nodes.
- **Checkpointing** — a long job can resume from the last completed node instead of restarting; pairs with the async job model (Section 9).

### 4.2 Graph state (illustrative)

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

class Scene(TypedDict):
    index: int
    narration: str
    duration_s: float
    image_prompt: str
    search_terms: list[str]
    asset_path: str | None
    asset_kind: Literal["ai_image", "stock_clip", None]
    retries: int

class VideoState(TypedDict):
    vision: str
    aspect_ratio: str
    settings: dict          # image_provider, voice, max_images, ...
    keys: dict              # llm_provider/key, image_provider/key
    script: str
    script_quality: float
    script_iters: int
    scenes: list[Scene]
    audio_path: str | None
    mp4_url: str | None
    error: str | None
```

### 4.3 Nodes

| Node | Responsibility | Loops / branches |
|------|----------------|------------------|
| `refine_script` | LLM refines the vision into a short-form script | → `critique_script` |
| `critique_script` | LLM scores the script; decides rewrite vs proceed | if score < threshold and iters < N → `refine_script`; else → `segment_scenes` |
| `segment_scenes` | LLM splits script into scenes (narration + duration + search terms) | → `make_prompts` |
| `make_prompts` | LLM writes a cinematic image prompt per scene | → `gen_visual` (fan-out per scene) |
| `gen_visual` | Generate AI image (user key) for a scene | on success → `verify_visual`; on failure → `fallback_stock` |
| `verify_visual` | (Optional) vision model checks image matches prompt | match → next scene; mismatch and retries < cap → `gen_visual`; exhausted → `fallback_stock` |
| `fallback_stock` | Fetch Pexels clip for the scene | → next scene |
| `generate_voice` | TTS over final narration | → `assemble` |
| `assemble` | FFmpeg + MoviePy: visuals + audio + subtitles + music + Ken Burns | → `upload` |
| `upload` | Push MP4 to storage, set `mp4_url` | → `END` |

### 4.4 Conditional edges (pseudocode)

```python
def route_after_critique(state: VideoState):
    if state["script_quality"] < QUALITY_THRESHOLD and state["script_iters"] < MAX_SCRIPT_ITERS:
        return "refine_script"
    return "segment_scenes"

def route_after_gen(state, scene):
    if scene["asset_path"] is None:                       # generation failed
        return "fallback_stock"
    return "verify_visual"                                 # or "next" if verify disabled

def route_after_verify(state, scene):
    if scene_matches(scene):
        return "next"
    if scene["retries"] < MAX_IMAGE_RETRIES:
        scene["retries"] += 1
        return "gen_visual"
    return "fallback_stock"
```

### 4.5 Scene-level parallelism
Scenes are independent, so `gen_visual` → `verify_visual` → `fallback_stock` can fan out across scenes (LangGraph parallel branches or an async map) and join before `generate_voice`/`assemble`. Respect the `max_images` cap and per-provider rate limits.

---

## 5. Image generation (primary new component)

### 5.1 Provider tiers

A multi-provider image module sits in front of MoneyPrinterTurbo's material-acquisition stage.

| Provider | Model | Approx cost / image (1K) | Notes |
|----------|-------|--------------------------|-------|
| Google / Gemini | Nano Banana 2 (Gemini 3.1 Flash Image) | ~$0.067 (batch ~$0.034) | Same key can serve script + images |
| fal.ai | FLUX 2 Pro | ~$0.055 | Provider-agnostic default, best price/quality |
| OpenAI | GPT Image 1 Mini / 1 | ~$0.005–0.04 | Same key can serve script + images |
| None | Pexels / Pixabay stock | Free | Real video clips, no key needed |

### 5.2 Resolution order

1. User selected a paid provider and supplied a valid key → generate AI image from the scene's prompt.
2. Generation fails (API error, rate limit, etc.) → fall back to Pexels stock footage.
3. No provider selected → use Pexels stock footage directly.

### 5.3 Resolver (pseudocode)

```python
def get_scene_visual(scene, settings, user_keys):
    provider = settings.image_provider  # "gemini" | "fal" | "openai" | "none"

    if provider != "none":
        key = user_keys.get(provider)
        if key:
            try:
                return generate_ai_image(provider, scene.image_prompt, key)
            except Exception as e:
                log.warning(f"AI gen failed ({provider}), falling back to stock: {e}")

    return fetch_pexels_clip(scene.search_terms)  # existing MPT logic
```

### 5.4 Cost guardrails
- Max images per video (configurable cap, e.g. 30) to prevent runaway spend on a user's key.
- Default to 1K resolution (sufficient for fast-moving Shorts); premium resolutions are an explicit opt-in.

---

## 6. API key handling & UX

### 6.1 Principles
- Surface **one simple choice**, not a menu of model names.
- Label providers by the **service the user already has an account with**, not by model jargon.
- Default to **free stock footage** so the product works with zero input.
- The user pays — make billing transparent.

### 6.2 Image source selector
A single dropdown:
- Google / Gemini — Nano Banana ("if you use Google AI")
- fal.ai — FLUX ("best general quality")
- OpenAI — GPT Image ("if you use ChatGPT")
- None — use free stock footage *(default)*

When a paid provider is selected, reveal a key field below it. When "None" is selected, hide the key field and show a "free stock footage" note.

### 6.3 Key reuse rule
- If the chosen image provider **matches** the provider used for the script LLM (e.g. script = Gemini, images = Gemini), **auto-fill** the existing key and show a "Reused" indicator — but keep the field **editable**.
- If the image provider **differs** from the script provider, present an **empty** key field for a fresh key.
- A key typed into the image field **always takes precedence** over the reused script key.
- Rationale: convenience for the common case, while allowing power users to use a separate billing key for image spend.

### 6.4 Validation & transparency
- Validate key format per provider; show a friendly error on mismatch (e.g. "this doesn't look like a Gemini key").
- Show a cost hint near the field (e.g. "≈ $0.05–0.07 per scene image, billed to your account").
- Provide a "Where do I get a key?" link per provider.

---

## 7. Technical considerations

### 7.1 Mixing AI stills with stock video
AI-generated images are static; Pexels returns video clips. To avoid a jarring mix:
- Apply a Ken Burns (zoom/pan) effect to AI stills via FFmpeg `zoompan` or MoviePy so they have motion comparable to stock clips.

### 7.2 Aspect ratio
- Generate or crop all visuals to the target ratio **before** assembly (9:16 for Shorts/TikTok/Reels, 16:9 for standard YouTube) to avoid inconsistent letterboxing.

### 7.3 Subtitles, music
- Reuse MoneyPrinterTurbo's existing subtitle generation (Whisper) and background-music support.

---

## 8. Architecture & deployment

### 8.1 Stack
- **Backend:** Python + FastAPI (from MoneyPrinterTurbo).
- **Orchestration:** LangGraph (graph state, conditional routing, retry loops, checkpointing).
- **Job queue:** Celery + Redis (async job execution; Redis shared with LangGraph checkpointing and rate limiting).
- **Auth & secrets:** user accounts with login; API keys stored with KMS envelope encryption (AWS/GCP KMS or HashiCorp Vault).
- **Assembly:** FFmpeg + MoviePy.
- **Subtitles:** Whisper.
- **Web UI:** MoneyPrinterTurbo's interface (extended with the image-source selector).
- **Containerization:** Docker (provided by MoneyPrinterTurbo).

### 8.2 LLM / TTS providers (user-supplied keys)
- LLM: OpenAI, Anthropic, Gemini, DeepSeek, Ollama (local), Qwen, and others (MPT-supported).
- TTS (MPT built-ins, routed by voice-name prefix):
  - **Edge TTS** — default, free, no key required (Microsoft/Azure neural voices via the `edge_tts` library). Used out of the box.
  - **Azure Speech SDK** — optional premium voices via the official Azure Cognitive Services Speech SDK; requires a user-supplied Azure Speech key.
  - **SiliconFlow TTS** — optional REST provider (`FunAudioLLM/CosyVoice2-0.5B`), with controllable voice, speed (0.25–4.0), gain, and 32 kHz sample rate; requires a user-supplied key.
- Voiceover key handling mirrors the image-source pattern: Edge TTS works with no key; Azure/SiliconFlow are opt-in with the user's own key.
- Synchronized SRT subtitles are generated from TTS word-boundary timing.

### 8.3 Job execution model
Video generation is long-running (tens of seconds to several minutes per video), so the work must never block an HTTP request.

- **Non-blocking API (always):** `POST /api/videos` enqueues a job and returns a `job_id` immediately. Clients poll `GET /api/videos/{job_id}` for status and the final MP4 URL.
- **Queue: Celery + Redis from launch.** Chosen over in-process background tasks because this is a multi-user product: FFmpeg rendering is CPU-heavy, so concurrent jobs need a controlled worker pool, jobs must survive server restarts/redeploys, and workers should scale independently of the web server. Celery also gives retries and queue visibility.
- **Shared Redis instance:** the same Redis used as the Celery broker/result backend also backs LangGraph checkpointing (Section 4.1), so resumable jobs and the queue reuse one piece of infrastructure rather than two.
- **Concurrency control:** cap worker concurrency to available CPU to keep FFmpeg renders from starving each other.

### 8.4 Storage & hosting
- **Output storage:** Cloudflare R2 (first 10 GB free) — returns a signed downloadable MP4 URL.
- **Hosting:** $5–10/month VPS (e.g. Railway, Render, DigitalOcean droplet).

### 8.5 Estimated platform cost
- Infrastructure: ~$5–10/month VPS, plus a small managed Redis instance (or self-hosted Redis on the same VPS at no extra cost for the MVP).
- AI generation: $0 to the platform (billed to users' own keys).

### 8.6 Authentication, key storage & rate limiting

**Authentication (login-first).** Users must create an account and log in before generating videos. Accounts give a stable identity to attach encrypted API keys to and to rate-limit against. (Auth method — email/password, OAuth, or magic link — is an implementation choice; any standard session/JWT approach is fine.)

**API key storage (encrypted, per account).** User-supplied keys (LLM, image, TTS) are persisted per account so users don't re-enter them each session.
- **Envelope encryption via a KMS** (AWS KMS, Google Cloud KMS, or HashiCorp Vault) is the target: each stored key is encrypted with a data key that is itself encrypted by a master key which never leaves the KMS. The database never holds anything decryptable on its own — a DB compromise alone does not expose keys.
- Avoid a single app-wide master key in env vars: if the server is compromised, all users' keys leak. Since keys bill to users' own accounts, a leak is direct financial harm to them.
- Keys are decrypted only in memory at job execution time, never written to logs, error traces, or job records.

**Rate limiting & abuse protection (two layers, backed by the shared Redis).**
- **Request-rate limiting** — per-user and per-IP caps on API calls to stop scripted hammering (e.g. SlowAPI for FastAPI, Redis-backed).
- **Job-rate / concurrency limiting** — the layer that protects *platform* cost (CPU, worker slots, R2 storage), since every job consumes those even though AI spend is on the user's key. Enforce: max concurrent jobs per user, max jobs per hour/day, and bounds on video length / scene count (`max_images`).
- The Redis instance already used for Celery and LangGraph checkpointing also backs the rate limiter — one piece of infrastructure, three uses.

---

## 9. API surface (MVP)

### `POST /api/videos`
Submit a generation job.

**Request (illustrative):**
```json
{
  "vision": "A 60-second explainer on why bees matter to agriculture",
  "aspect_ratio": "9:16",
  "settings": {
    "image_provider": "gemini",
    "voice": "en-US-AvaNeural",
    "max_images": 15
  },
  "keys": {
    "llm_provider": "gemini",
    "llm_key": "AIza...",
    "image_provider": "gemini",
    "image_key": "AIza..."
  }
}
```

**Response:**
```json
{ "job_id": "abc123", "status": "queued" }
```

### `GET /api/videos/{job_id}`
Poll job status and retrieve the result.

**Response:**
```json
{
  "job_id": "abc123",
  "status": "completed",
  "progress": 100,
  "mp4_url": "https://.../abc123.mp4"
}
```

---

## 10. Build phases

### Phase 1 — MVP (≈8–12 days)
- Fork MoneyPrinterTurbo and run it end-to-end.
- Add user accounts + login; gate video generation behind auth (Section 8.6).
- Persist user API keys with KMS envelope encryption; per-user + per-IP rate limiting and per-user job/concurrency caps (Section 8.6).
- Stand up Celery + Redis; make the API non-blocking (enqueue job → return `job_id` → poll for status).
- Persist user API keys with KMS envelope encryption; per-user + per-IP rate limiting and per-user job/concurrency caps (Section 8.6).
- Wrap MPT stage functions as LangGraph nodes; build the graph with state, conditional fallback routing, and the per-scene retry loop (Section 4).
- Build the multi-provider image module + Pexels fallback (Section 5).
- Add the image-source selector + key reuse logic (Section 6).
- Wire output upload to Cloudflare R2.
- Deploy to a VPS.

### Phase 2 — Polish
- Script-critic loop and optional vision-based image verification (Section 4.1).
- LangGraph checkpointing wired to the async job model for resumable jobs.
- Ken Burns on AI stills; aspect-ratio handling.
- Cost guardrails and key validation hardening.
- Premium resolution / model toggles.

### Phase 3 — Optional extensions
- Publishing automation and scheduling on top of the pipeline API (auto-post finished MP4s to social platforms).
- Scene-level parallel fan-out tuning and provider rate-limit handling.

---

## 11. Open questions
- Authentication method: email/password vs. OAuth vs. magic link (login-first is decided — see Section 8.6; only the mechanism is open).
- KMS choice: AWS KMS vs. Google Cloud KMS vs. self-hosted HashiCorp Vault (depends on final hosting provider).