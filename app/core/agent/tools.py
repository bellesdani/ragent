import re

from typing import Optional
from datetime import datetime
from zoneinfo import ZoneInfo
from app.core.agent.entities import AgentDeps
from pydantic_ai import Agent, ModelRetry, RunContext
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchText


def _extract_ip_addresses(ip_addresses: Optional[str]) -> list[str]:
    if not ip_addresses:
        return []

    ip_pattern = (
        r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
    )
    return list(dict.fromkeys(re.findall(ip_pattern, ip_addresses)))


def _extract_mac_addresses(mac_addresses: Optional[str]) -> list[str]:
    if not mac_addresses:
        return []

    mac_pattern = (
        r"\b(?:[0-9A-Fa-f]{2}([-:]))(?:[0-9A-Fa-f]{2}\1){4}[0-9A-Fa-f]{2}\b"
        r"|\b[0-9A-Fa-f]{12}\b"
    )
    return list(dict.fromkeys(match.group(0) for match in re.finditer(mac_pattern, mac_addresses)))


def _extract_serial_numbers(serial_numbers: Optional[str]) -> list[str]:
    if not serial_numbers:
        return []

    return list(
        dict.fromkeys(
            sn.strip().upper()
            for sn in re.split(r"[\s,;]+", serial_numbers)
            if sn.strip()
        )
    )


def _extract_article_ids(article_ids: Optional[str]) -> list[str]:
    if not article_ids:
        return []

    article_id_pattern = r"\b[A-Z]{3}\d{6}\b"
    return list(dict.fromkeys(match.group(0) for match in re.finditer(article_id_pattern, article_ids)))


def _build_articles_filter(article_ids: list[str]) -> Filter | None:
    filter_conditions = []
    if article_ids:
        filter_conditions.append(
            FieldCondition(
                key="article.id",
                match=MatchAny(any=article_ids)
            )
        )
    if not filter_conditions:
        return None

    return Filter(must=filter_conditions)


def _build_devices_filter(ip_addresses: list[str], mac_addresses: list[str], serial_numbers: list[str]) -> Filter | None:
    filter_conditions = []
    if ip_addresses:
        filter_conditions.append(
            FieldCondition(
                key="device.ip_addresses",
                match=MatchAny(any=ip_addresses)
            )
        )
    if mac_addresses:
        filter_conditions.append(
            FieldCondition(
                key="device.mac_addresses",
                match=MatchAny(any=mac_addresses)
            )
        )
    if serial_numbers:
        filter_conditions.append(
            FieldCondition(
                key="device.serial_number",
                match=MatchAny(any=serial_numbers)
            )
        )

    if not filter_conditions:
        return None

    return Filter(must=filter_conditions)



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
                        key="employee.department",
                        match=MatchText(text=department),
                    )
                ]
            )

        # Llamada a la base de conocimiento
        retrieval = await context.deps.retriever.retrieve(
            limit=10,
            query=query,
            source_id="employees",
            query_filter=qdrant_filter,
        )

        # Prepara la información para la generación de la respuesta
        if retrieval.documents:
            lines = []
            lines.append(f"Consulta reescrita: {retrieval.query}")
            lines.append(f"Resultado de la búsqueda:")
            for index, document in enumerate(retrieval.documents, start=1):
                # Aquí yo voy a ignorar el content que utilizo para el embedding y me voy a basar solo en metadata,
                #  ya que metadata guarda el json de cada empleado y puede tener datos más interesantes que el propio content
                lines.append(f" [{index}] Fuente: {document.id}")
                lines.append(f" [{index}] Contenido: {document.metadata}")
            if retrieval.last_data_update:
                lines.append(f"Datos actualizados a fecha de: {retrieval.last_data_update}")
            return "\n".join(lines)

        return "No se encontró evidencia relevante en las fuentes solicitadas."


def register_devices_retrieval_tool(agent: Agent[AgentDeps, str]) -> None:
    @agent.tool
    async def search_devices(
        context: RunContext[AgentDeps],
        query: str,
        ip_addresses: Optional[str] = None,
        mac_addresses: Optional[str] = None,
        serial_numbers: Optional[str] = None,
    ) -> str:
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

            ip_addresses: Texto que contiene una o varias IPs cuando el usuario quiere buscar dispositivos por dirección IP exacta.
                Si no hay direcciones IP claras, deja este parámetro como None.

            mac_addresses: Texto que contiene una o varias direcciones MAC cuando el usuario quiere buscar dispositivos por dirección MAC exacta.
                Si no hay direcciones MAC claras, deja este parámetro como None.

            serial_numbers: Texto que contiene uno o varios números de serie cuando el usuario quiere buscar dispositivos por número de serie exacto.
                Si no hay números de serie claros, deja este parámetro como None.
            
        """
        # Validación de la lista de IPs
        valid_ips = _extract_ip_addresses(ip_addresses)
        if ip_addresses and not valid_ips:
            raise ModelRetry("No se han encontrado direcciones IP válidas en la consulta.")

        # Validación de la lista de direcciones MAC
        valid_macs = _extract_mac_addresses(mac_addresses)
        if mac_addresses and not valid_macs:
            raise ModelRetry("No se han encontrado direcciones MAC válidas en la consulta.")

        # Normalización de la lista de números de serie
        normalized_serial_numbers = _extract_serial_numbers(serial_numbers)
        if serial_numbers and not normalized_serial_numbers:
            raise ModelRetry("No se han encontrado números de serie válidos en la consulta.")

        # Construimos el filtro de Qdrant
        qdrant_filter = _build_devices_filter(valid_ips, valid_macs, normalized_serial_numbers)
        exact_values = valid_ips + valid_macs + normalized_serial_numbers
        retrieval_query = ", ".join(exact_values) if exact_values else query
        retrieval_limit = max(len(valid_ips), len(valid_macs), len(normalized_serial_numbers), 1) if qdrant_filter else 5

        # Llamada a la base de conocimiento
        retrieval = await context.deps.retriever.retrieve(
            limit=retrieval_limit,
            query=retrieval_query,
            source_id="devices",
            query_filter=qdrant_filter,
        )

        # Prepara la información para la generación de la respuesta
        if retrieval.documents:
            lines = []
            lines.append(f"Consulta reescrita: {retrieval.query}")
            lines.append(f"Resultado de la búsqueda:")
            for index, document in enumerate(retrieval.documents, start=1):
                # Aquí yo voy a ignorar el content que utilizo para el embedding y me voy a basar solo en metadata,
                #  ya que metadata guarda el json de cada device y puede tener datos más interesantes que el propio content
                lines.append(f" [{index}] Fuente: {document.id}")
                lines.append(f" [{index}] Contenido: {document.metadata}")
                # lines.append(f" [{index}] Contenido: {document.content}")
            if retrieval.last_data_update:
                lines.append(f"Datos actualizados a fecha de: {retrieval.last_data_update}")
            return "\n".join(lines)
        else:
            return "No se encontró evidencia relevante en las fuentes solicitadas."


def register_articles_retrieval_tool(agent: Agent[AgentDeps, str]) -> None:
    @agent.tool
    async def search_articles(
        context: RunContext[AgentDeps],
        query: str,
        article_ids: Optional[str] = None,
    ) -> str:
        """
        Esta herramienta permite recuperar de una colección de Qdrant información sobre los artículos guardados en el ERP.
        Tanto aquellos artículos que producimos en Equipe Cerámicas (VEN) como los que adquirimos a otras empresas.
        Permite obtener información específica sobre dicho artículo como por ejemplo:
         - Descripción del artículo
         - Fecha y usuario de alta.
         - Características específicas de azulejos que producimos en Equipe como el adn, formato, etc.

        Args:
            query: Consulta autónoma, concreta y optimizada para búsqueda.
                Si la pregunta del usuario depende del historial, debes incorporar el contexto relevante.
                No uses referencias ambiguas como "su", "ese artículo", "el anterior" o "esa información".

            article_ids: Texto que contiene uno o varios identificadores de artículos cuando el usuario quiere buscar artículos por id.
                Si no hay identificadores de artículo claros, deja este parámetro como None.
                Los identificadores de artículo empiezan con un prefijo de tres letras seguidos por seis números.
                Algunos ejemplos son: VEN02384, PRO008842, MTO486248, COM985421 
            
        """
        # Validación de la lista de IDs
        valid_ids = _extract_article_ids(article_ids)
        if valid_ids and not valid_ids:
            raise ModelRetry("No se han encontrado identificadores de artículo válidos en la consulta.")

        # Construimos el filtro de Qdrant
        qdrant_filter = _build_articles_filter(valid_ids)
        exact_values = valid_ids
        retrieval_query = ", ".join(exact_values) if exact_values else query
        retrieval_limit = max(len(valid_ids), 1) if qdrant_filter else 50

        # Llamada a la base de conocimiento
        retrieval = await context.deps.retriever.retrieve(
            limit=retrieval_limit,
            query=retrieval_query,
            source_id="articles",
            query_filter=qdrant_filter,
        )

        # Prepara la información para la generación de la respuesta
        if retrieval.documents:
            lines = []
            lines.append(f"Consulta reescrita: {retrieval.query}")
            lines.append(f"Resultado de la búsqueda:")
            for index, document in enumerate(retrieval.documents, start=1):
                # Aquí yo voy a ignorar el content que utilizo para el embedding y me voy a basar solo en metadata,
                #  ya que metadata guarda el json de cada device y puede tener datos más interesantes que el propio content
                lines.append(f" [{index}] Fuente: {document.id}")
                lines.append(f" [{index}] Contenido: {document.metadata}")
                # lines.append(f" [{index}] Contenido: {document.content}")
            if retrieval.last_data_update:
                lines.append(f"Datos actualizados a fecha de: {retrieval.last_data_update}")
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
            limit=5,
            query=query,
            source_id="manuals",
        )

        # Prepara la información para la generación de la respuesta
        if retrieval.documents:
            lines = []
            lines.append(f"Consulta reescrita: {retrieval.query}")
            lines.append(f"Resultado de la búsqueda:")
            for index, document in enumerate(retrieval.documents, start=1):
                lines.append(f" [{index}] Fuente: {document.id}")
                lines.append(f" [{index}] Contenido: {document.content}")
                lines.append(f" [{index}] Metadata: {document.metadata}")
            if retrieval.last_data_update:
                lines.append(f"Datos actualizados a fecha de: {retrieval.last_data_update}")
            return "\n".join(lines)
        else:
            return "No se encontró evidencia relevante en las fuentes solicitadas."


def register_tickets_retrieval_tool(agent: Agent[AgentDeps, str]) -> None:
    @agent.tool
    async def search_tickets(context: RunContext[AgentDeps], query: str) -> str:
        """
        Esta herramienta permite recuperar de una colección de Qdrant información sobre las incidencias (tickets) registrados a través de HelpDesk.
        Permite obtener posibles soluciones a problemas ya resueltos o buscar información sobre indidencias registradas.
        También puede ser una fuente de información alternativa a los manuales si en esta fuente de datos no se encontrase información.

        Args:
            query: Consulta autónoma, concreta y optimizada para búsqueda.
            Si la pregunta del usuario depende del historial, debes incorporar el contexto relevante.
            No uses referencias ambiguas como "el anterior", "esa herramienta" o "ese programa".
        """
        # Llamada a la base de conocimiento
        retrieval = await context.deps.retriever.retrieve(
            limit=3,
            query=query,
            source_id="tickets",
        )

        # Prepara la información para la generación de la respuesta
        if retrieval.documents:
            lines = []
            lines.append(f"Consulta reescrita: {retrieval.query}")
            lines.append(f"Resultado de la búsqueda:")
            for index, document in enumerate(retrieval.documents, start=1):
                # Para el caso de los tickets, vamos a eliminar el histórico de artículos porque añade mucha basura
                if "articles" in document.metadata:
                    document.metadata["articles"] = []
                # Para cada ticket elegido, devolvemos identificador, contenido y metadatos
                lines.append(f" [{index}] Fuente: {document.id}")
                lines.append(f" [{index}] Contenido: {document.content}")
                lines.append(f" [{index}] Metadata: {document.metadata}")
            if retrieval.last_data_update:
                lines.append(f"Datos actualizados a fecha de: {retrieval.last_data_update}")
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
        Esta herramienta devuelve el día y la hora exacta actual.
        """
        return datetime.now(ZoneInfo("Europe/Madrid"))
    
