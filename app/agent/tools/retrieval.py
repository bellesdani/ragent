from __future__ import annotations

from pydantic_ai import Agent, RunContext

from app.agent.deps import AgentDeps


def register_retrieval_tool(agent: Agent[AgentDeps, str]) -> None:
    @agent.tool
    async def search_sources(ctx: RunContext[AgentDeps], query: str, source_ids: list[str] | None = None) -> str:
        """
        Busca informacion en Qdrant usando una o varias colecciones.

        Args:
            query: Consulta a buscar en la base de conocimiento.
            source_ids: Lista opcional de colecciones a consultar. Si se omite, busca en todas.
        """
        retrieval = await ctx.deps.retriever.retrieve_from_sources(
            query=query,
            messages=ctx.deps.messages,
            source_ids=source_ids,
        )
        if not retrieval.documents:
            return "No se encontro evidencia relevante en las fuentes solicitadas."

        lines = [f"Consulta reescrita: {retrieval.query}"]
        for index, document in enumerate(retrieval.documents, start=1):
            source = (
                document.metadata.get("source_name")
                or document.metadata.get("collection")
                or document.id
            )
            lines.append(f"[{index}] Fuente: {source}")
            lines.append(f"[{index}] Score: {document.score:.4f}")
            lines.append(f"[{index}] Texto: {document.text}")
        return "\n".join(lines)
