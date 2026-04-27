from app.core.entities import AgentDeps
from pydantic_ai import Agent, ModelRetry, RunContext


def register_employees_retrieval_tool(agent: Agent[AgentDeps, str]) -> None:
    @agent.tool
    async def search_employees(context: RunContext[AgentDeps], query: str) -> str:
        """
        Esta herramienta permite recuperar de una colección de Qdrant información sobre los empleados y compañeros de la empresa.
        Permite obtener información básica del empleado, como por ejemplo:
         - Nombre y apellidos.
         - Departamento.
        Permite obtener la información de contacto, como por ejemplo:
         - Números de teléfono y extensión.
         - Direcciones de correo electrónico.
         
        Args:
            query: Consulta autónoma, concreta y optimizada para búsqueda.
            Si la pregunta del usuario depende del historial, debes incorporar el contexto relevante.
            No uses referencias ambiguas como "su", "ese equipo", "el anterior" o "esa persona".
        """
        # Llamada a la base de conocimiento
        retrieval = await context.deps.retriever.retrieve(
            query=query,
            source_ids=["employees"],
        )

        # Prepara la información para la generación de la respuesta
        if retrieval.documents:
            lines = [f"Consulta usada: {retrieval.query}"]
            for index, document in enumerate(retrieval.documents, start=1):
                lines.append(f"[{index}] Fuente: {document.id}")
                # lines.append(f"[{index}] Texto: {document.text}")
                # lines.append(f"[{index}] Score: {document.score}")
                lines.append(f"[{index}] Texto: {document.metadata}")
            return "\n".join(lines)
        else:
            return "No se encontró evidencia relevante en las fuentes solicitadas."


def register_devices_retrieval_tool(agent: Agent[AgentDeps, str]) -> None:
    @agent.tool
    async def search_devices(context: RunContext[AgentDeps], query: str) -> str:
        """
        Esta herramienta permite recuperar de una colección de Qdrant información sobre los equipos, y ordenadores de la empresa.
        Permite obtener información específica sobre los dispositivos como por ejemplo:
         - Nombre del host
         - Sistema operativo
         - Número de serie
         - Direcciones IP
         - Direcciones MAC
         - VLANs
         - Proveedor
         - Modelo
         - Hardware del dispositivo

        En algunos casos, también muestra la información del usuario que tiene asgindado, pero es posible que sea necesario realizar la búsqueda por el correo o nombre de usuario, para ello puede que sea necesario iterar con la herramienta de empleados

        Args:
            query: Consulta autónoma, concreta y optimizada para búsqueda.
            Si la pregunta del usuario depende del historial, debes incorporar el contexto relevante.
            No uses referencias ambiguas como "su", "ese equipo", "el anterior" o "esa persona".
        """
        # Llamada a la base de conocimiento
        retrieval = await context.deps.retriever.retrieve(
            query=query,
            source_ids=["devices"],
        )

        # Prepara la información para la generación de la respuesta
        if retrieval.documents:
            lines = [f"Consulta usada: {retrieval.query}"]
            for index, document in enumerate(retrieval.documents, start=1):
                lines.append(f"[{index}] Fuente: {document.id}")
                # lines.append(f"[{index}] Texto: {document.text}")
                # lines.append(f"[{index}] Score: {document.score}")
                lines.append(f"[{index}] Texto: {document.metadata}")
            return "\n".join(lines)
        else:
            return "No se encontró evidencia relevante en las fuentes solicitadas."


def register_calculator_tool(agent: Agent[AgentDeps, str]) -> None:
    @agent.tool_plain
    def calculator(expression: str) -> str:
        """
        Evalúa una expresión aritmética simple.

        Args:
            expression: Expresión matemática usando números, paréntesis y operadores básicos.
        """
        # Validación de la expresión
        if not expression.strip():
            raise ModelRetry("La expresión no puede estar vacía.")
        
        if any(char not in "0123456789+-*/()., %\t\n" for char in expression):
            raise ModelRetry("La expresión contiene caracteres no permitidos.")
        
        # Normalización y ejecución de las operaciones artiméticas
        normalized = expression.replace(",", ".")
        try:
            result = eval(normalized, {"__builtins__": {}}, {})
        except Exception as exc:
            raise ModelRetry(f"No se pudo evaluar la expresión: {exc}") from exc
        return str(result)
