from __future__ import annotations

from app.prompts.loader import load_prompt


class PromptBuilder:
    def build_agent_system_prompt(self, template_path: str) -> str:
        return load_prompt(template_path)
