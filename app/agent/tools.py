from __future__ import annotations


from app.agent.types import AgentDeps
from pydantic_ai import Agent, ModelRetry, RunContext


def register_agent_tools(agent: Agent[AgentDeps, str], enable_retrieval: bool) -> None:
    if enable_retrieval:

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
                return "No se encontró evidencia relevante en las fuentes solicitadas."

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

    @agent.tool_plain
    def calculator(expression: str) -> str:
        """
        Evalua una expresion aritmetica simple.

        Args:
            expression: Expresion matematica usando numeros, parentesis y operadores basicos.
        """
        if not expression.strip():
            raise ModelRetry("La expresion no puede estar vacia.")
        if any(char not in "0123456789+-*/()., %\t\n" for char in expression):
            raise ModelRetry("La expresion contiene caracteres no permitidos.")
        normalized = expression.replace(",", ".")
        try:
            result = eval(normalized, {"__builtins__": {}}, {})
        except Exception as exc:  # pragma: no cover - depende de la expresion del usuario
            raise ModelRetry(f"No se pudo evaluar la expresion: {exc}") from exc
        return str(result)
