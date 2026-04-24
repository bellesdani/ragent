from __future__ import annotations

from pathlib import Path
from string import Template
from functools import lru_cache


PROMPTS_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def load_prompt(relative_path: str) -> str:
    prompt_path = PROMPTS_DIR / relative_path
    return prompt_path.read_text(encoding="utf-8").strip()


