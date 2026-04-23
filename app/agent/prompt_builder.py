from __future__ import annotations

import json
from collections.abc import Callable

from app.api.schemas.openai import RetrievalDocument
from app.prompts import load_prompt, render_prompt


class PromptBuilder:
    def build_agent_system_prompt(self, template_path: str) -> str:
        return load_prompt(template_path)

    def build_planner_prompt(self, agent_id: str, sources: list[dict[str, str]]) -> str:
        return render_prompt(
            "planner/retrieval_planner.md",
            agent_id=agent_id,
            sources_json=json.dumps(sources, ensure_ascii=False),
        )

    def build_answer_system_prompt(
        self,
        base_system_prompt: str,
        documents: list[RetrievalDocument],
        should_search: bool,
        format_metadata: Callable[[dict[str, object]], str],
    ) -> str:
        system_parts = [base_system_prompt]
        if should_search:
            if documents:
                citations = []
                for index, document in enumerate(documents, start=1):
                    source = document.metadata.get("source_name") or document.metadata.get("collection") or document.id
                    metadata = format_metadata(document.metadata)
                    citation_parts = [f"[{index}] {source}"]
                    if metadata:
                        citation_parts.append(f"Metadata: {metadata}")
                    citation_parts.append(document.text)
                    citations.append("\n".join(citation_parts))
                prefix = load_prompt("answer/with_context_prefix.md")
                system_parts.append(prefix + "\n" + "\n\n".join(citations))
            else:
                system_parts.append(load_prompt("answer/no_evidence.md"))
        else:
            system_parts.append(load_prompt("answer/without_context.md"))
        return "\n\n".join(system_parts)
