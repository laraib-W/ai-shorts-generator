import json
import re

from loguru import logger

from app.pipeline.constants import MAX_SCRIPT_ITERS, QUALITY_THRESHOLD
from app.pipeline.llm_context import llm_override
from app.pipeline.state import VideoState
from app.services.llm import _generate_response, generate_script

SCRIPT_SYSTEM_PROMPT = """\
# Role: Short-Form Video Script Writer

## Context:
You write voiceover narration for short-form video (YouTube Shorts, TikTok, Reels).
The script will be read aloud over visuals at approximately 150 words per minute.

## Rules:
1. Open with a hook — a surprising fact, bold claim, or urgent question that grabs attention immediately.
2. Write for the ear, not the eye: short punchy sentences, natural rhythm, conversational tone. Avoid complex vocabulary.
3. Each paragraph must represent a visually distinct scene for later video editing. Separate paragraphs with a single blank line.
4. Build a simple arc: hook → build curiosity/tension → deliver the payoff → end with a call to action or a seamless loop.
5. Target 30-90 seconds of narration (roughly 75-225 words total).
6. Output ONLY the spoken narration. No markdown, no bold text, no titles, no stage directions, no visual cues (e.g., do not write "Scene 1" or "[Visual: ...]").
7. Do not reference this prompt or mention paragraphs/structure. Just write the raw script.
8. Respond in the same language as the video subject or user brief.
""".strip()

CRITIQUE_PROMPT = """You are a video script critic for short-form content (YouTube Shorts, TikTok, Reels).

Rate the following script on a scale of 1-10 based on:
- Clarity: Is it easy to follow?
- Pacing: Does it fit a 30-90 second video?
- Engagement: Does it hook the viewer in the first few seconds?
- Narration quality: Does it sound natural when read aloud?

Script:
\"\"\"
{script}
\"\"\"

Respond ONLY with valid JSON, no other text:
{{"score": <number 1-10>, "feedback": "<one sentence explaining what to improve>"}}"""

REWRITE_PROMPT = """You are a video script writer for short-form content (YouTube Shorts, TikTok, Reels).

Rewrite the following script based on the feedback. Return ONLY the improved script text, nothing else.

Original script:
\"\"\"
{script}
\"\"\"

Feedback: {feedback}"""


def refine_script(state: VideoState) -> dict:
    vision = state["vision"]
    script_iters = state.get("script_iters", 0)

    with llm_override(state.get("keys", {})):
        if script_iters == 0:
            logger.info(f"generating initial script from vision: {vision[:100]}...")
            script = generate_script(
                video_subject=vision,
                language=state["settings"].get("video_language", ""),
                paragraph_number=state["settings"].get("paragraph_number", 1),
                custom_system_prompt=SCRIPT_SYSTEM_PROMPT,
            )
        else:
            feedback = state.get("_critique_feedback", "")
            logger.info(f"rewriting script (iter {script_iters}), feedback: {feedback}")
            prompt = REWRITE_PROMPT.format(script=state["script"], feedback=feedback)
            script = _generate_response(prompt)

    return {"script": script}


def critique_script(state: VideoState) -> dict:
    script = state["script"]
    script_iters = state.get("script_iters", 0)

    logger.info(f"critiquing script (iter {script_iters})...")
    prompt = CRITIQUE_PROMPT.format(script=script)

    with llm_override(state.get("keys", {})):
        try:
            raw = _generate_response(prompt)
            raw = raw.strip()
            # Handle cases where LLM wraps JSON in markdown code blocks
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            result = json.loads(raw)
            score = float(result.get("score", 0))
            feedback = result.get("feedback", "")
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"failed to parse critique response, assuming passing score: {e}")
            score = QUALITY_THRESHOLD
            feedback = ""

    logger.info(f"script quality: {score}/10 (threshold: {QUALITY_THRESHOLD})")

    return {
        "script_quality": score,
        "script_iters": script_iters + 1,
        "_critique_feedback": feedback,
    }
