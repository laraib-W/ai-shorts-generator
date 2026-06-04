# AI Video Generator

A web service that converts a short text idea into a fully generated short-form video (MP4) for YouTube Shorts, TikTok, and Instagram Reels. The system refines your idea into a script, splits it into scenes, generates visuals and voiceover, and assembles everything into a downloadable video.

Users bring their own API keys for AI services, keeping platform costs near zero.

## How It Works

1. Submit a vision (short text describing your video idea)
2. The pipeline refines it into a script, segments scenes, generates visuals and voiceover
3. FFmpeg assembles everything into a captioned, narrated MP4
4. Download the finished video

## Stack

- **Backend:** Python, FastAPI
- **Orchestration:** LangGraph (stateful pipeline with retries and fallback routing)
- **Job Queue:** Celery + Redis
- **Assembly:** FFmpeg + MoviePy
- **Subtitles:** Whisper
- **Storage:** Cloudflare R2

## Image Providers

Users choose an image source per video (or default to free stock footage):

| Provider | Notes |
|----------|-------|
| Google / Gemini | Uses your existing Google AI key |
| fal.ai (FLUX) | Best general quality |
| OpenAI (GPT Image) | Uses your existing OpenAI key |
| None (default) | Free stock footage from Pexels |

## Getting Started

### Prerequisites

- Python 3.11+
- Redis
- FFmpeg
- ImageMagick

### Setup

```bash
git clone https://github.com/your-org/ai-video-generator.git
cd ai-video-generator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your configuration
```

### Run

```bash
# Start Redis (if not already running)
redis-server

# Start Celery worker
celery -A app.worker worker --loglevel=info

# Start the API server
uvicorn app.main:app --reload
```

## API

### Create a video

```
POST /api/videos
```

```json
{
  "vision": "A 60-second explainer on why bees matter to agriculture",
  "aspect_ratio": "9:16",
  "settings": {
    "image_provider": "gemini",
    "voice": "en-US-AvaNeural"
  }
}
```

Returns `{ "job_id": "abc123", "status": "queued" }`.

### Check status

```
GET /api/videos/{job_id}
```

Returns status, progress, and the MP4 URL when complete.

## License

MIT
