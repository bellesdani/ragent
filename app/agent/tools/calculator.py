from __future__ import annotations

from pydantic_ai import Agent, ModelRetry

from app.agent.deps import AgentDeps


def register_calculator_tool(agent: Agent[AgentDeps, str]) -> None:
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
