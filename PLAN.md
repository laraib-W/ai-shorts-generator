# AI Video Generator — Implementation Plan (Fork-Based)

## Context
Building an AI text-to-video generator by **forking MoneyPrinterTurbo (MPT)** and replacing its linear pipeline with **LangGraph orchestration**. MPT already provides all the stage implementations we need (LLM, TTS, FFmpeg/MoviePy assembly, Pexels stock footage, subtitles). Our job is to:

1. Replace MPT's linear `task.py` orchestrator with a LangGraph StateGraph
2. Add the multi-provider AI image generation module (new — MPT only has stock footage)
3. Add Celery + Redis async job queue (MPT has basic BackgroundTasks + optional Redis)
4. Add auth, API key encryption, rate limiting
5. Wire output upload to Cloudflare R2

**What we keep from MPT (untouched or lightly modified):**
- `app/services/llm.py` — Multi-provider LLM client (13+ providers)
- `app/services/voice.py` — TTS (Edge TTS, Azure, Gemini, SiliconFlow, MiMo)
- `app/services/video.py` — MoviePy video assembly, transitions, subtitle rendering
- `app/services/material.py` — Pexels/Pixabay stock footage fetching
- `app/services/subtitle.py` — SRT subtitle generation (Edge timestamps or Whisper)
- `app/config/config.py` — TOML-based config system
- `app/models/schema.py` — Existing Pydantic models (extend, don't replace)
- `resource/` — Fonts, background music
- `Dockerfile`, `docker-compose.yml` — Base Docker setup (extend)

**What we replace/add:**
- `app/services/task.py` → `app/pipeline/` (LangGraph graph replaces linear orchestrator)
- New: `app/services/image.py` (multi-provider AI image generation)
- New: `app/services/storage.py` (Cloudflare R2 upload)
- New: `app/pipeline/` (LangGraph state, nodes, edges, graph)
- New: `app/worker.py` + `app/tasks.py` (Celery async jobs)
- New: Auth system, API key encryption, rate limiting

---

## Step 1: Fork & Setup
Clone MPT into this repo, verify it runs, add LangGraph + new dependencies.

### Actions:
- Clone MoneyPrinterTurbo into this repo (preserve git history or fresh copy)
- Verify the existing app runs: `python main.py` starts FastAPI on :8080
- Verify Streamlit UI runs: `streamlit run webui/Main.py`
- Add new dependencies to `pyproject.toml`: `langgraph`, `celery[redis]`, `boto3`, `python-jose[cryptography]`, `passlib[bcrypt]`, `sqlalchemy`, `slowapi`, `cryptography`
- Update `.env.example` with new env vars (R2, KMS, app secret)

### Outcome:
- MPT runs end-to-end (stock footage video from a topic)
- New dependencies installed alongside existing ones

---

## Step 2: LangGraph State & Types
Define the typed state that flows through the LangGraph pipeline. This maps MPT's existing data flow into a formal state object.

### Files to create:
- `app/pipeline/__init__.py`
- `app/pipeline/state.py`

### Key types:
```python
class Scene(TypedDict):
    index: int
    narration: str
    duration_s: float
    image_prompt: str
    search_terms: list[str]
    asset_path: str | None
    asset_kind: Literal["ai_image", "stock_clip"] | None
    retries: int

class VideoState(TypedDict):
    # Input
    vision: str
    aspect_ratio: str          # "9:16", "16:9", "1:1"
    settings: dict             # image_provider, voice_name, max_images, ...
    keys: dict                 # llm_provider/key, image_provider/key
    task_id: str
    task_dir: str              # MPT's per-task storage directory

    # Script stage
    script: str
    script_quality: float
    script_iters: int

    # Scene stage
    scenes: list[Scene]
    current_scene_index: int

    # Audio/subtitle stage
    audio_path: str | None
    subtitle_path: str | None

    # Assembly stage
    mp4_path: str | None
    mp4_url: str | None

    # Error handling
    error: str | None
```

### Constants:
```python
QUALITY_THRESHOLD = 7.0       # Script quality score (1-10)
MAX_SCRIPT_ITERS = 3          # Max script rewrite loops
MAX_IMAGE_RETRIES = 2         # Max image gen retries per scene
MAX_IMAGES_DEFAULT = 30       # Cost guardrail
```

### Outcome:
- Clean typed state that bridges MPT's existing data model with LangGraph

---

## Step 3: Pipeline Nodes — Script Refinement & Critique Loop
Build the first LangGraph nodes. These wrap MPT's existing `llm.generate_script()` and add the new critic loop.

### Files to create:
- `app/pipeline/nodes/__init__.py`
- `app/pipeline/nodes/script.py`
- `app/pipeline/edges.py`

### Node: `refine_script`
- Calls MPT's `app/services/llm.generate_script()` (or `_generate_response()` with a custom prompt)
- Takes `vision` from state, writes `script` to state

### Node: `critique_script`
- New LLM call: scores the script on clarity, pacing, engagement (1-10)
- Writes `script_quality` and increments `script_iters`

### Edge: `route_after_critique()`
```python
def route_after_critique(state: VideoState) -> str:
    if state["script_quality"] < QUALITY_THRESHOLD and state["script_iters"] < MAX_SCRIPT_ITERS:
        return "refine_script"
    return "segment_scenes"
```

### LangGraph concepts:
- Stateful nodes mutating VideoState
- Conditional edge routing (critic loop)
- Bounded iteration

### Outcome:
- Vision → refined script with quality-gated rewrites

---

## Step 4: Pipeline Nodes — Scene Segmentation & Image Prompt Generation
LLM splits the script into scenes. These extend MPT's existing `llm.generate_terms()` approach.

### Files to create:
- `app/pipeline/nodes/scenes.py`

### Node: `segment_scenes`
- LLM call: split script into Scene list with narration, duration_s, search_terms per scene
- Populates `state["scenes"]`

### Node: `make_prompts`
- LLM call: write a cinematic image prompt per scene (for AI image generation)
- Populates `scene["image_prompt"]` for each scene

### Outcome:
- Script → list of Scenes with narration, duration, search terms, and image prompts

---

## Step 5: Pipeline Nodes — Visual Generation with Retry & Fallback
The most complex LangGraph routing. Per-scene loop with AI image → verify → stock fallback.

### Files to create:
- `app/services/image.py` — **New**: multi-provider AI image generation
- `app/pipeline/nodes/visuals.py`
- Update `app/pipeline/edges.py`

### Service: `app/services/image.py` (NEW)
```python
def generate_image(provider: str, key: str, prompt: str, aspect_ratio: str) -> str:
    """Generate AI image. Returns local file path. Supports gemini, fal, openai."""
```

- **Gemini**: google-generativeai SDK (Imagen / Gemini Flash image gen)
- **fal.ai**: httpx REST call to FLUX 2 Pro
- **OpenAI**: openai SDK (GPT Image 1)
- Returns downloaded image path in task_dir

### Node: `gen_visual`
- If image_provider != "none" and key exists: call `generate_image()`
- On success: set `scene["asset_path"]`, `scene["asset_kind"] = "ai_image"`
- On failure: set `scene["asset_path"] = None`
- Advance `current_scene_index`

### Node: `verify_visual` (Phase 2 — stub for now)
- Optional vision model check: does the image match the prompt?
- For MVP: auto-pass

### Node: `fallback_stock`
- Calls MPT's existing `material.search_videos_pexels()` / `material.download_videos()`
- Sets `scene["asset_path"]`, `scene["asset_kind"] = "stock_clip"`

### Edges:
```python
def route_after_gen(state: VideoState) -> str:
    scene = state["scenes"][state["current_scene_index"]]
    if scene["asset_path"] is None:
        return "fallback_stock"
    return "verify_visual"  # or "next_scene" if verify disabled

def route_after_verify(state: VideoState) -> str:
    scene = state["scenes"][state["current_scene_index"]]
    if scene_matches(scene):             # MVP: always True
        return "check_scenes_done"
    if scene["retries"] < MAX_IMAGE_RETRIES:
        scene["retries"] += 1
        return "gen_visual"
    return "fallback_stock"

def check_scenes_done(state: VideoState) -> str:
    if state["current_scene_index"] >= len(state["scenes"]) - 1:
        return "generate_voice"
    return "gen_visual"  # next scene
```

### LangGraph concepts:
- Per-scene retry loop (bounded)
- Conditional fallback routing (AI → verify → stock)
- Scene iteration via state index

### Outcome:
- Each scene gets a visual (AI-generated or stock), with automatic fallback on failure

---

## Step 6: Pipeline Nodes — Voiceover & Assembly
Wrap MPT's existing TTS and video assembly as LangGraph nodes.

### Files to create:
- `app/pipeline/nodes/audio.py`
- `app/pipeline/nodes/assembly.py`

### Node: `generate_voice`
- Calls MPT's `voice.tts()` with full narration text and voice_name from settings
- Calls MPT's `subtitle.generate_subtitle()` for SRT
- Writes `audio_path` and `subtitle_path` to state

### Node: `assemble`
- Calls MPT's `video.generate_video()` / `video.concatenate_clips()`
- Passes scene asset paths, audio, subtitles, background music
- For AI stills: apply Ken Burns (zoom/pan) via MPT's existing MoviePy effects
- Writes `mp4_path` to state

### Outcome:
- Full pipeline produces an MP4 from vision text (end-to-end)

---

## Step 7: Pipeline Nodes — Upload & Output

### Files to create:
- `app/services/storage.py` — **New**: Cloudflare R2 upload
- `app/pipeline/nodes/output.py`

### Service: `app/services/storage.py`
```python
def upload_to_r2(file_path: str, task_id: str) -> str:
    """Upload MP4 to R2, return signed URL. Falls back to local file serving in dev."""
```
- Uses boto3 S3-compatible client pointed at R2 endpoint
- Dev mode: serve from local `storage/tasks/{task_id}/` via FastAPI static files

### Node: `upload`
- Calls `upload_to_r2()`, writes `mp4_url` to state

### Outcome:
- Final MP4 uploaded, downloadable URL returned

---

## Step 8: LangGraph — Assemble the Full Graph
Wire all nodes and edges into the complete StateGraph. This is the centerpiece.

### Files to create:
- `app/pipeline/graph.py`

### Graph structure:
```
START → refine_script → critique_script
                            ↓
              [conditional: loop back or proceed]
                            ↓
                      segment_scenes → make_prompts
                                          ↓
                                    gen_visual (scene loop)
                                     ↓          ↓
                              verify_visual   fallback_stock
                                     ↓          ↓
                              check_scenes_done ←┘
                                     ↓
                              [more scenes? → gen_visual]
                              [all done? ↓]
                              generate_voice → assemble → upload → END
```

### Implementation:
```python
from langgraph.graph import StateGraph, END

builder = StateGraph(VideoState)

# Add nodes
builder.add_node("refine_script", refine_script)
builder.add_node("critique_script", critique_script)
builder.add_node("segment_scenes", segment_scenes)
builder.add_node("make_prompts", make_prompts)
builder.add_node("gen_visual", gen_visual)
builder.add_node("verify_visual", verify_visual)
builder.add_node("fallback_stock", fallback_stock)
builder.add_node("check_scenes_done", check_scenes_done)
builder.add_node("generate_voice", generate_voice)
builder.add_node("assemble", assemble)
builder.add_node("upload", upload)

# Edges
builder.set_entry_point("refine_script")
builder.add_edge("refine_script", "critique_script")
builder.add_conditional_edges("critique_script", route_after_critique)
builder.add_edge("segment_scenes", "make_prompts")
builder.add_edge("make_prompts", "gen_visual")
builder.add_conditional_edges("gen_visual", route_after_gen)
builder.add_conditional_edges("verify_visual", route_after_verify)
builder.add_edge("fallback_stock", "check_scenes_done")
builder.add_conditional_edges("check_scenes_done", route_check_scenes)
builder.add_edge("generate_voice", "assemble")
builder.add_edge("assemble", "upload")
builder.add_edge("upload", END)

video_graph = builder.compile()
```

### Outcome:
- `video_graph.invoke(initial_state)` runs the full pipeline
- Can test with: `python -m app.pipeline.graph` (CLI smoke test)

---

## Step 9: API Integration
Update MPT's existing FastAPI endpoints to use the LangGraph pipeline instead of the linear `task.py`.

### Files to modify:
- `app/controllers/v1/video.py` — Wire `POST /v1/videos` to invoke `video_graph`
- `app/models/schema.py` — Add/extend request model with `vision`, `keys`, `image_provider` fields
- `app/router.py` — Add new routes if needed

### New endpoint (alongside existing):
```
POST /api/videos     — New LangGraph-based generation (vision → video)
GET  /api/videos/{job_id} — Poll status (sync first, async later)
```

### Keep existing MPT endpoints working:
```
POST /v1/videos      — Original MPT flow (still useful for testing)
GET  /v1/tasks/{id}  — Original task status
```

### Outcome:
- New API endpoint runs the LangGraph pipeline
- Old MPT endpoints still work for comparison/fallback

---

## Step 10: Celery + Redis Async Job Queue
Make video generation non-blocking. MPT already has optional Redis support — extend it.

### Files to create:
- `app/worker.py` — Celery app config, Redis broker/backend
- `app/tasks.py` — `generate_video_task` Celery task wrapping `video_graph.invoke()`

### Files to modify:
- `app/controllers/v1/video.py` — `POST /api/videos` enqueues Celery task, returns job_id
- `app/config/config.py` — Add Celery/Redis config (reuse existing `redis_host`/`redis_port`)

### Outcome:
- Non-blocking API: POST → job_id → poll GET for status/progress/mp4_url
- Celery worker runs independently, respects concurrency caps

---

## Step 11: Auth & User Accounts
Add user authentication. MPT has no auth — everything is open.

### Files to create:
- `app/services/auth.py` — JWT token creation/validation, password hashing
- `app/models/user.py` — User SQLAlchemy model
- `app/database.py` — SQLAlchemy setup (SQLite for MVP)
- `app/controllers/v1/auth.py` — Register, login, me endpoints
- `app/dependencies.py` — `get_current_user` FastAPI dependency

### Files to modify:
- `app/router.py` — Register auth routes
- `app/controllers/v1/video.py` — Gate video endpoints behind auth

### Outcome:
- Users must register/login before generating videos
- JWT token auth on all video endpoints

---

## Step 12: API Key Encryption & Storage
Persist user API keys securely so they don't re-enter them each session.

### Files to create:
- `app/services/encryption.py` — Fernet symmetric encryption (dev), interface ready for KMS
- `app/models/api_keys.py` — Encrypted key storage model (per user, per provider)
- `app/controllers/v1/keys.py` — CRUD endpoints for user API keys

### Outcome:
- Users save/update/delete their API keys per provider
- Keys encrypted at rest, decrypted only at job execution time

---

## Step 13: Rate Limiting
Protect the platform. MPT has `max_concurrent_tasks` / `max_queued_tasks` — extend with per-user limits.

### Files to create:
- `app/middleware/rate_limit.py` — Redis-backed per-user + per-IP rate limiting (SlowAPI or custom)

### Files to modify:
- `app/tasks.py` — Per-user job concurrency caps
- `app/asgi.py` — Add rate limit middleware

### Outcome:
- Request-rate and job-rate limiting enforced

---

## Step 14: UI — Image Source Selector
Extend MPT's Streamlit UI with the image provider dropdown and key fields.

### Files to modify:
- `webui/Main.py` — Add image source dropdown (Gemini / fal.ai / OpenAI / None), key field with reuse logic, cost hints

### Outcome:
- Users can select AI image provider or default to free stock footage
- Key reuse from LLM provider when providers match

---

## Step 15: Docker & Deployment

### Files to modify:
- `Dockerfile` — Add new dependencies, keep FFmpeg
- `docker-compose.yml` — Add Celery worker service, ensure Redis service exists

### Files to create:
- `scripts/start.sh` — Entrypoint for web + worker

### Outcome:
- `docker-compose up` runs FastAPI + Celery worker + Redis
- Ready for VPS deployment

---

## Verification Plan
1. **Step 1:** MPT runs unmodified — generate a stock footage video via API
2. **Step 8:** `python -m app.pipeline.graph` with a test vision produces an MP4
3. **Step 9:** `POST /api/videos` with vision text returns a video URL
4. **Step 10:** POST returns job_id immediately, GET polls until complete
5. **Step 11:** Unauthenticated requests rejected with 401
6. **Step 15:** `docker-compose up` and full flow works in containers

---

## Dependencies to Add (on top of MPT's existing)
```
langgraph                        # Pipeline orchestration
celery[redis]                    # Async job queue
boto3                            # Cloudflare R2 upload (S3-compatible)
python-jose[cryptography]        # JWT auth
passlib[bcrypt]                  # Password hashing
sqlalchemy                       # User/key database
slowapi                          # Rate limiting
cryptography                     # Fernet encryption for API keys
```

## MPT Existing Dependencies (kept as-is)
```
fastapi, uvicorn                 # API server
moviepy                          # Video assembly
edge-tts                         # Free TTS
openai, google-generativeai      # LLM providers
faster-whisper                   # Subtitle transcription
redis                            # State management
pydub                            # Audio processing
litellm                          # LLM gateway
loguru                           # Logging
streamlit                        # Web UI
```
