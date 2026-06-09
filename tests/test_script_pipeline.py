"""Test script for Step 3: Script Refinement & Critique Loop.

Runs the refine_script and critique_script nodes manually,
then shows the routing decision from route_after_critique.
"""

from app.pipeline.nodes.script import refine_script, critique_script
from app.pipeline.edges import route_after_critique

# Simulate an initial VideoState
state = {
    "vision": "A 45-second explainer on why bees are critical to global agriculture",
    "aspect_ratio": "9:16",
    "settings": {
        "video_language": "en-US",
        "paragraph_number": 3,
    },
    "keys": {
        "llm_provider": "gemini",
        "llm_key": "",  # empty = falls through to config.toml key
    },
    "task_id": "test-001",
    "task_dir": "/tmp/test-task",
    "script": "",
    "script_quality": 0.0,
    "script_iters": 0,
    "scenes": [],
    "current_scene_index": 0,
    "audio_path": None,
    "subtitle_path": None,
    "mp4_path": None,
    "mp4_url": None,
    "error": None,
}

print("=" * 60)
print("STEP 1: Generating initial script from vision...")
print("=" * 60)
result = refine_script(state)
state.update(result)
print(f"\nGenerated script ({len(state['script'].split())} words):")
print("-" * 40)
print(state["script"])

print("\n" + "=" * 60)
print("STEP 2: Critiquing the script...")
print("=" * 60)
result = critique_script(state)
state.update(result)
print(f"\nScore: {state['script_quality']}/10")
print(f"Iterations: {state['script_iters']}")
print(f"Feedback: {state.get('_critique_feedback', 'none')}")

print("\n" + "=" * 60)
print("STEP 3: Routing decision...")
print("=" * 60)
next_node = route_after_critique(state)
print(f"Next node: {next_node}")

if next_node == "refine_script":
    print("\nScript quality below threshold — would loop back for rewrite.")
    print("Running rewrite...")
    result = refine_script(state)
    state.update(result)
    print(f"\nRewritten script ({len(state['script'].split())} words):")
    print("-" * 40)
    print(state["script"])

    print("\nRe-critiquing...")
    result = critique_script(state)
    state.update(result)
    print(f"New score: {state['script_quality']}/10")
    print(f"Iterations: {state['script_iters']}")
    next_node = route_after_critique(state)
    print(f"Next node: {next_node}")
else:
    print("\nScript passed quality threshold — would proceed to segment_scenes.")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
