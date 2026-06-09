import threading
from contextlib import contextmanager

from app.config import config

_lock = threading.Lock()


@contextmanager
def llm_override(keys: dict):
    """Thread-safe context manager that temporarily injects user-supplied keys
    into MPT's global config so that _generate_response() picks them up.

    Usage:
        with llm_override(state["keys"]):
            result = _generate_response(prompt)

    Keys dict expected format:
        {
            "llm_provider": "gemini",
            "llm_key": "AIza...",
            "llm_base_url": "",       # optional
            "llm_model_name": "",     # optional
        }
    """
    provider = keys.get("llm_provider", "")
    api_key = keys.get("llm_key", "")
    base_url = keys.get("llm_base_url", "")
    model_name = keys.get("llm_model_name", "")

    if not provider:
        yield
        return

    config_keys = {
        "llm_provider": provider,
    }
    if api_key:
        config_keys[f"{provider}_api_key"] = api_key
    if base_url:
        config_keys[f"{provider}_base_url"] = base_url
    if model_name:
        config_keys[f"{provider}_model_name"] = model_name

    with _lock:
        saved = {}
        for k, v in config_keys.items():
            saved[k] = config.app.get(k)
            config.app[k] = v
        try:
            yield
        finally:
            for k, original in saved.items():
                if original is None:
                    config.app.pop(k, None)
                else:
                    config.app[k] = original
