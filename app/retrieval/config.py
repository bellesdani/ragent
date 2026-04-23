from __future__ import annotations

from dataclasses import dataclass


DEFAULT_TOP_K = 15
DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_CONTEXT_MAX_CHARS = 12000
DEFAULT_PAYLOAD_TEXT_KEYS = ("text", "content", "chunk", "page_content")


@dataclass(frozen=True)
class KnowledgeSource:
    id: str
    name: str
    description: str
    collection: str
    vector_name: str | None = None


DEFAULT_SEARCH_SOURCES = (
    KnowledgeSource(
        id="devices",
        name="Devices",
        description="Informacion sobre los dispositivos de la empresa, tales como servidores y equipos de usuario y de planta.",
        collection="devices",
    ),
    KnowledgeSource(
        id="employees",
        name="Employees",
        description="Informacion sobre los empleados de la empresa y su contacto corporativo, como correo electronico, telefono y extension.",
        collection="employees",
    ),
)
