from typing import Literal, TypedDict

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
    # --- Input ---
    vision: str
    aspect_ratio: str  # "9:16", "16:9", "1:1" — maps to MPT's VideoAspect
    settings: dict  # image_provider, voice_name, max_images, subtitle, font, bgm, ...
    keys: dict  # llm_provider, llm_key, image_provider, image_key

    # Matches MPT's per-task storage convention (storage/tasks/{task_id}/)
    task_id: str
    task_dir: str

    # --- Script stage ---
    script: str
    script_quality: float
    script_iters: int

    # --- Scene stage ---
    scenes: list[Scene]
    current_scene_index: int

    # --- Audio/subtitle stage ---
    audio_path: str | None
    subtitle_path: str | None

    # --- Assembly stage ---
    mp4_path: str | None
    mp4_url: str | None

    # --- Error handling ---
    error: str | None
