from typing import TYPE_CHECKING
from dataclasses import dataclass
from app.core.chat.entities import ChatMessage

if TYPE_CHECKING:
    from app.core.knowledge_source.service import KnowledgeSourceService


@dataclass(frozen=True)
class AgentDefinition:
    agent_id: str
    name: str
    description: str
    backend_base_url: str
    backend_api_key: str
    backend_chat_model: str
    system_prompt: str
    enable_tools: bool
    public: bool


@dataclass
class AgentDeps:
    knowledge_service: "KnowledgeSourceService"
    messages: list[ChatMessage]
