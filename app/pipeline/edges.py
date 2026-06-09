from app.pipeline.constants import MAX_SCRIPT_ITERS, QUALITY_THRESHOLD
from app.pipeline.state import VideoState


def route_after_critique(state: VideoState) -> str:
    if (
        state["script_quality"] < QUALITY_THRESHOLD
        and state["script_iters"] < MAX_SCRIPT_ITERS
    ):
        return "refine_script"
    return "segment_scenes"
