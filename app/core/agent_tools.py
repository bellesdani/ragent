import re

from typing import Optional
from datetime import datetime
from app.core.entities import AgentDeps
from pydantic_ai import Agent, ModelRetry, RunContext
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchText


def register_employees_retrieval_tool(agent: Agent[AgentDeps, str]) -> None:
    @agent.tool
    async def search_employees(context: RunContext[AgentDeps], query: str, department: Optional[str] = None) -> str:
        """
        Recupera información sobre empleados y compañeros de la empresa.

        Permite obtener información básica del empleado:
         - Nombre y apellidos.
         - Departamento.

        Permite obtener información de contacto:
         - Números de teléfono y extensión.
         - Direcciones de correo electrónico.

        Args:
            query: Consulta autónoma, concreta y optimizada para búsqueda.
                Si la pregunta del usuario depende del historial, incorpora el contexto relevante.
                No uses referencias ambiguas como "su", "ese equipo", "el anterior" o "esa persona".

            department: Departamento por el que filtrar cuando el usuario lo mencione explícita o implícitamente. 
                    Usa uno de estos valores base: "Informática", "Mantenimiento", "Comunicación", "Sostenibilidad", "Producción", "Finanzas", "Administración", "Comercial", "Suministros", "Diseño", "Prevención", "Logística", "Recursos Humanos", "Marketing", "Muestras", "Calidad", "Taller", "Gerencia"
                    Si no hay un departamento claro, deja este parámetro como None.
        """

        # Prepara el filtro de Qdrant por departamento y lo valida si es necesario
        qdrant_filter = None
        allowed_departments = set(["Informática", 
                                   "Mantenimiento", 
                                   "Comunicación",
                                   "Sostenibilidad",
                                   "Producción",
                                   "Finanzas",
                                   "Administración",
                                   "Comercial",
                                   "Suministros",
                                   "Diseño",
                                   "Prevención",
                                   "Logística",
                                   "Recursos Humanos",
                                   "Marketing",
                                   "Muestras",
                                   "Calidad",
                                   "Taller",
                                   "Gerencia",
                                   ])
        if department and department in allowed_departments:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.department",
                        match=MatchText(text=department),
                    )
                ]
            )

        # Llamada a la base de conocimiento
        retrieval = await context.deps.retriever.retrieve(
            query=query,
            source_ids=["employees"],
            query_filter=qdrant_filter,
        )

        # Prepara la información para la generación de la respuesta
        if retrieval.documents:
            lines = [f"Consulta usada: {retrieval.query}"]
            if department:
                lines.append(f"Filtro aplicado: department = {department}")
            for index, document in enumerate(retrieval.documents, start=1):
                # Aquí yo voy a ignorar el content que utilizo para el embedding y me voy a basar solo en metadata,
                #  ya que metadata guarda el json de cada empleado y puede tener datos más interesantes que el propio content
                lines.append("")
                lines.append(f"[{index}] Fuente: {document.id}")
                lines.append(f"[{index}] Contenido: {document.metadata}")
            return "\n".join(lines)

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
                lines.append("")
                lines.append(f"[{index}] Fuente: {document.id}")
                lines.append(f"[{index}] Contenido: {document.metadata}")
                # lines.append(f"[{index}] Contenido: {document.text}")
            return "\n".join(lines)
        else:
            return "No se encontró evidencia relevante en las fuentes solicitadas."


def register_ip_retrieval_tool(agent: Agent[AgentDeps, str]) -> None:
    @agent.tool
    async def search_devices_by_ip_address(context: RunContext[AgentDeps], ip_addresses: str) -> str:
        """
        Esta herramienta permite recuperar de una colección de Qdrant información sobre los equipos, y ordenadores de la empresa a partir de una o varias direcciones IP.
        Permite obtener información específica sobre esos dispositivos de forma más fiable que la búsqueda genérica en los dispositivos.

        Args:
            ip_addresses: Texto que contiene una o varias IPs (ej: "192.168.1.10, 10.0.0.5")
        """
        # Validación de la lista de IPs
        ip_pattern = (
            r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
        )
        valid_ips = list(dict.fromkeys(re.findall(ip_pattern, ip_addresses)))
        if not valid_ips:
            raise ModelRetry("No se han encontrado direcciones IP válidas en la consulta.")

        # Construimos el filtro de Qdrant
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.ips",
                    match=MatchAny(any=valid_ips)
                )
            ]
        )

        # Llamada a la base de conocimiento
        retrieval = await context.deps.retriever.retrieve(
            query=", ".join(valid_ips),
            source_ids=["devices"],
            query_filter=qdrant_filter,
        )

        # Prepara la información para la generación de la respuesta
        if retrieval.documents:
            lines = [f"Consulta usada: {retrieval.query}"]
            for index, document in enumerate(retrieval.documents, start=1):
                # Aquí yo voy a ignorar el content que utilizo para el embedding y me voy a basar solo en metadata,
                #  ya que metadata guarda el json de cada device y puede tener datos más interesantes que el propio content
                lines.append(f"")
                lines.append(f"[{index}] Fuente: {document.id}")
                lines.append(f"[{index}] Contenido: {document.metadata}")
                # lines.append(f"[{index}] Contenido: {document.text}")
            return "\n".join(lines)
        else:
            return "No se encontró evidencia relevante en las fuentes solicitadas."


def register_mac_retrieval_tool(agent: Agent[AgentDeps, str]) -> None:
    @agent.tool
    async def search_devices_by_mac_address(context: RunContext[AgentDeps], mac_addresses: str) -> str:
        """
        Esta herramienta permite recuperar de una colección de Qdrant información sobre los equipos, y ordenadores de la empresa a partir de una o varias direcciones MAC.
        Permite obtener información específica sobre esos dispositivos de forma más fiable que la búsqueda genérica en los dispositivos.

        Args:
            mac_addresses: Texto que contiene una o varias direcciones MAC (ej: "D4:E9:8A:D9:90:2C, 3C:0A:F3:0F:B9:21")
        """
        # Validación de la lista de direcciones MAC
        mac_pattern = (
            r"\b(?:[0-9A-Fa-f]{2}([-:]))(?:[0-9A-Fa-f]{2}\1){4}[0-9A-Fa-f]{2}\b"
            r"|\b[0-9A-Fa-f]{12}\b"
        )
        valid_macs = list(dict.fromkeys(match.group(0) for match in re.finditer(mac_pattern, mac_addresses)))
        if not valid_macs:
            raise ModelRetry("No se han encontrado direcciones IP válidas en la consulta.")

        # Construimos el filtro de Qdrant
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.mac_addresses",
                    match=MatchAny(any=valid_macs)
                )
            ]
        )

        # Llamada a la base de conocimiento
        retrieval = await context.deps.retriever.retrieve(
            query=", ".join(valid_macs),
            source_ids=["devices"],
            query_filter=qdrant_filter,
        )

        # Prepara la información para la generación de la respuesta
        if retrieval.documents:
            lines = [f"Consulta usada: {retrieval.query}"]
            for index, document in enumerate(retrieval.documents, start=1):
                # Aquí yo voy a ignorar el content que utilizo para el embedding y me voy a basar solo en metadata,
                #  ya que metadata guarda el json de cada device y puede tener datos más interesantes que el propio content
                lines.append(f"")
                lines.append(f"[{index}] Fuente: {document.id}")
                lines.append(f"[{index}] Contenido: {document.metadata}")
                # lines.append(f"[{index}] Contenido: {document.text}")
            return "\n".join(lines)
        else:
            return "No se encontró evidencia relevante en las fuentes solicitadas."


def register_serial_number_retrieval_tool(agent: Agent[AgentDeps, str]) -> None:
    @agent.tool
    async def search_devices_by_serial_number(context: RunContext[AgentDeps], serial_numbers: str) -> str:
        """
        Esta herramienta permite recuperar de una colección de Qdrant información sobre los equipos, y ordenadores de la empresa a partir de uno o varios números de serie.
        Permite obtener información específica sobre esos dispositivos de forma más fiable que la búsqueda genérica en los dispositivos.

        Args:
            serial_numbers: Texto que contiene uno o varios números de serie (ej: "17JWE88D0XXC25D0080, 451444HH1NYX8")
        """
        # Normalización de la lista de números de serie
        normalized_serial_numbers = list(
            dict.fromkeys(
                sn.strip().upper()
                for sn in re.split(r"[\s,;]+", serial_numbers)
                if sn.strip()
            )
        )

        if not normalized_serial_numbers:
            raise ModelRetry("No se han encontrado números de serie válidos en la consulta.")
        
        # Construimos el filtro de Qdrant
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.serial_number",
                    match=MatchAny(any=normalized_serial_numbers)
                )
            ]
        )

        # Llamada a la base de conocimiento
        retrieval = await context.deps.retriever.retrieve(
            query=", ".join(normalized_serial_numbers),
            source_ids=["devices"],
            query_filter=qdrant_filter,
        )

        # Prepara la información para la generación de la respuesta
        if retrieval.documents:
            lines = [f"Consulta usada: {retrieval.query}"]
            for index, document in enumerate(retrieval.documents, start=1):
                # Aquí yo voy a ignorar el content que utilizo para el embedding y me voy a basar solo en metadata,
                #  ya que metadata guarda el json de cada device y puede tener datos más interesantes que el propio content
                lines.append(f"")
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
                lines.append("")
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


def register_datetime_tool(agent: Agent[AgentDeps, str]) -> None:
    @agent.tool_plain
    def get_current_time() -> datetime:
        """
        Devuelve la hora exacta actual.
        """
        return datetime.now()
    
