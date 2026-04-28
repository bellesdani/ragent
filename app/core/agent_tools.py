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
                # Aquí yo voy a ignorar el content que utilizo para el embedding y me voy a basar solo en metadata,
                #  ya que metadata guarda el json de cada empleado y puede tener datos más interesantes que el propio content
                lines.append(f"[{index}] Fuente: {document.id}")
                lines.append(f"[{index}] Contenido: {document.metadata}")
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
                # Aquí yo voy a ignorar el content que utilizo para el embedding y me voy a basar solo en metadata,
                #  ya que metadata guarda el json de cada device y puede tener datos más interesantes que el propio content
                lines.append(f"[{index}] Fuente: {document.id}")
                lines.append(f"[{index}] Contenido: {document.metadata}")
                # lines.append(f"[{index}] Contenido: {document.text}")
            return "\n".join(lines)
        else:
            return "No se encontró evidencia relevante en las fuentes solicitadas."


def register_manuals_retrieval_tool(agent: Agent[AgentDeps, str]) -> None:
    @agent.tool
    async def search_manuals(context: RunContext[AgentDeps], query: str) -> str:
        """
        Esta herramienta permite recuperar de una colección de Qdrant información sobre manuales y operativas habituales registradas de la empresa.
        Algunos ejemplos son: de manuales del Forti, EtiPrinter, EtiPistola, e incluso del propio ERP (EKON):
         - Manual oficial del Forti, nuestro Firewall. 
         - Manual de uso de herramientas internas como EtiPrinter, EtiPistola, etc.
         - Manual de operativas habituales a través del propio ERP, llamado Ekon.
         - Manuales de gestión de dispositivos en RED desde el programa UniFi

        Args:
            query: Consulta autónoma, concreta y optimizada para búsqueda.
            Si la pregunta del usuario depende del historial, debes incorporar el contexto relevante.
            No uses referencias ambiguas como "el anterior", "esa herramienta" o "ese programa".
        """
        # Llamada a la base de conocimiento
        retrieval = await context.deps.retriever.retrieve(
            query=query,
            source_ids=["manuals"],
        )

        # Prepara la información para la generación de la respuesta
        if retrieval.documents:
            lines = [f"Consulta usada: {retrieval.query}"]
            for index, document in enumerate(retrieval.documents, start=1):
                lines.append(f"[{index}] Fuente: {document.id}")
                lines.append(f"[{index}] Contenido: {document.text}")
                lines.append(f"[{index}] Metadata: {document.metadata}")
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
