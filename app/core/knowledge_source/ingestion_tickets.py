import re

from html import unescape
from typing import Optional
from app.config import Settings
from qdrant_client import models
from app.core.agent.service import AgentService
from app.core.chat.entities import ChatResult, ChatMessage
from app.core.knowledge_source.ingestion_abc import KnowledgeSourceIngestor
from app.core.knowledge_source.entities import KnowledgeSourceDefinition, TicketArticleRow, TicketArticle, Ticket



class TicketsKnowledgeSourceIngestor(KnowledgeSourceIngestor):
    """
    Este ingestor prepara los tickets para su consulta como fuente de conocimiento. Utiliza:
     - La configuración base de ingesta (KnowledgeSourceIngestor)
     - El agente de resumen de tickets

    Funciones públicas:
     - Crear la fuente de conocimiento de tickets (create_knowledge_source).
     - Añadir datos de tickets a la fuente de conocimiento (upsert_knowledge_source_data).
    """

    def __init__(self, settings: Settings, knowledge_source: KnowledgeSourceDefinition, agent_service: AgentService) -> None:
        super().__init__(
            settings=settings,
            knowledge_source=knowledge_source,
        )
        self.agent_service = agent_service


    async def create_knowledge_source(self):
        # Si la colección no existe, la creamos
        if await self.qdrant_client.collection_exists(self.knowledge_source.collection_name):
            return False
        
        # Validamos que hemos configurado el nombre de los vectores en la definición
        dense_vector_name = self.knowledge_source.dense_vector_name
        sparse_vector_name = self.knowledge_source.sparse_vector_name
        assert dense_vector_name is not None and sparse_vector_name is not None

        # Creamos la colección
        await self.qdrant_client.create_collection(
            collection_name=self.knowledge_source.collection_name,
            vectors_config={
                dense_vector_name: models.VectorParams(
                    size=2048,
                    distance=models.Distance.COSINE,
                    on_disk=False,
                    hnsw_config=models.HnswConfigDiff(
                        m=24,
                        payload_m=24,
                        ef_construct=256,
                    ),
                    datatype=models.Datatype.FLOAT32,
                )
            },
            sparse_vectors_config={
                sparse_vector_name: models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=True,
                    ),
                    modifier=models.Modifier.IDF,
                )
            },
        )

        # Añadimos índices para optimizar los filtros
        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.title",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                lowercase=True,
                tokenizer=models.TokenizerType.WHITESPACE,
                phrase_matching=True
            )
        )
        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.number",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
            ),
        )

        return True


    async def upsert_knowledge_source_data(self, data):
        # Primero validamos el modelo de dato que recibimos
        # Esperamos un conjunto de filas de ticket - artículo
        ticket_articles = [
            TicketArticleRow.model_validate(item)
            for item in data
        ]

        # En segundo lugar, si no existe la colección en Qdrant, la creamos
        if not await self.qdrant_client.collection_exists(self.knowledge_source.collection_name):
            await self.create_knowledge_source()

        # En tercer lugar, agrupamos los artículos de los tickets en tickets
        #  Pasamos de tener entradas de la base de datos a datos agrupados
        tickets: dict[int, Ticket] = {}
        for ticket_article in ticket_articles:
            # Si no existe el ticket, lo añadimos
            if ticket_article.ticket_id not in tickets:
                tickets[ticket_article.ticket_id] = Ticket(
                    id = ticket_article.ticket_id,
                    group_id = ticket_article.ticket_group_id,
                    priority_id = ticket_article.ticket_priority_id,
                    state_id = ticket_article.ticket_state_id,
                    organization_id = ticket_article.ticket_organization_id,
                    number = ticket_article.ticket_number,
                    title = ticket_article.ticket_title,
                    created_at = ticket_article.ticket_created_at,
                    closed_at = ticket_article.ticket_closed_at,
                    customer_firstname = ticket_article.ticket_customer_firstname,
                    customer_lastname = ticket_article.ticket_customer_lastname,
                    customer_department = ticket_article.ticket_customer_department,
                    customer_email = ticket_article.ticket_customer_email,
                    creator_firstname = ticket_article.ticket_creator_firstname,
                    creator_lastname = ticket_article.ticket_creator_lastname,
                    creator_department = ticket_article.ticket_creator_department,
                    creator_email = ticket_article.ticket_creator_email,
                    articles = [
                        TicketArticle(
                            id = ticket_article.article_id,
                            from_email = ticket_article.article_from,
                            to_email = ticket_article.article_to,
                            subject = ticket_article.article_subject,
                            content_type = ticket_article.article_content_type,
                            body = ticket_article.article_body,
                            internal = ticket_article.article_internal,
                            created_at = ticket_article.article_created_at,
                            creator_firstname = ticket_article.article_creator_firstname,
                            creator_lastname = ticket_article.article_creator_lastname,
                            creator_department = ticket_article.article_creator_department,
                            creator_email = ticket_article.article_creator_email,
                        )
                    ]
                )
            
            # Si ya existe, solo añadimos el artículo
            else:
                tickets[ticket_article.ticket_id].articles.append(
                    TicketArticle(
                        id = ticket_article.article_id,
                        from_email = ticket_article.article_from,
                        to_email = ticket_article.article_to,
                        subject = ticket_article.article_subject,
                        content_type = ticket_article.article_content_type,
                        body = ticket_article.article_body,
                        internal = ticket_article.article_internal,
                        created_at = ticket_article.article_created_at,
                        creator_firstname = ticket_article.article_creator_firstname,
                        creator_lastname = ticket_article.article_creator_lastname,
                        creator_department = ticket_article.article_creator_department,
                        creator_email = ticket_article.article_creator_email,
                    )
                )

        # En cuarto lugar, preparamos el payload
        # Payload: 
        #  - Semantic_content: Texto para generar los embeddings de búsqueda semántica
        #  - Lexical_content: Texto para procesar con bm25 para la búsqueda léxica  
        #  - Metadata: Información relativa al ticket, como id, usuarios, etc
        payloads = []
        for ticket in tickets.values():
            metadata = self._build_ticket_metadata(ticket)
            lexical_content = self._build_lexical_content(ticket)
            semantic_content = await self._build_semantic_content(lexical_content)

            payload = {
                self.knowledge_source.payload_keys.lexical_content_key: lexical_content,
                self.knowledge_source.payload_keys.semantic_content_key: semantic_content,
                self.knowledge_source.payload_keys.metadata_key: metadata
            }
            payloads.append(payload)
                

        # En quinto lugar, definimos los puntos de Qdrant y antes validamos que hemos configurado los vectores
        # Point:
        #  - id: Identificador del punto
        #  - payload: Payload del punto
        #  - vector: Vectores para la búsqueda, densos para búsqueda semántica y dispersos para busqueda léxica 
        dense_vector_name = self.knowledge_source.dense_vector_name
        sparse_vector_name = self.knowledge_source.sparse_vector_name
        assert dense_vector_name is not None and sparse_vector_name is not None

        points: list[models.PointStruct] = []
        for payload in payloads:
            points.append(
                models.PointStruct(
                    id=payload[self.knowledge_source.payload_keys.metadata_key]["id"],
                    vector={
                        dense_vector_name: await self.embedding_client.create_embedding(
                            input_text=payload[self.knowledge_source.payload_keys.semantic_content_key],
                            model=self.settings.embedding_model
                        ),
                        sparse_vector_name: models.Document(
                            text=payload[self.knowledge_source.payload_keys.lexical_content_key],
                            model="Qdrant/bm25",
                        ),
                    },
                    payload=payload,
                )
            )

        # Finalmente, insertamos los puntos Qdrant 
        if len(points) > 0:
            await self.qdrant_client.upsert(
                collection_name=self.knowledge_source.collection_name,
                points=points,
            )
        return {
            "tickets": len(tickets),
            "points": len(points),
        }


    def _build_lexical_content(self, ticket: Ticket) -> str:
        lines :list[str] = []
        lines.append(f"Ticket: {ticket.title}")
        for article in ticket.articles:
                article_body_cleaned = self._clean_text(article.body)
                if article.from_email:
                    lines.append(f" - Mensaje de: {article.from_email}")
                if article.to_email:
                    lines.append(f" - Para: {article.to_email}")
                if article.subject:
                    lines.append(f" - Tema: {article.subject}")
                if article_body_cleaned:
                    lines.append(f" - Contenido: {article_body_cleaned}")
                lines.append("")
        return "\n".join(lines)    
    

    async def _build_semantic_content(self, content: str):
        result: ChatResult = await self.agent_service.complete_chat(
            model="Summarizer",
            messages=[
                ChatMessage(
                    content=content,
                    role='user',
                    name='',
                ),
            ],
            max_tokens=400,
            temperature=0.2,
        )

        return result.content
    
    
    def _build_ticket_metadata(self, ticket: Ticket):
        return ticket.model_dump(mode="json")
        

    def _clean_text(self, text: Optional[str]) -> str:
        if not text:
            return ""

        cleaned = text

        # Eliminar contenido no visible
        cleaned = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", cleaned)

        # Convertir saltos y bloques HTML habituales en saltos de línea
        cleaned = re.sub(r"(?i)<br\s*/?>", "\n", cleaned)
        cleaned = re.sub(r"(?i)</?(p|div|li|tr|table|tbody|thead|section|article|h[1-6])[^>]*>", "\n", cleaned)

        # Eliminar el resto de etiquetas HTML
        cleaned = re.sub(r"(?s)<[^>]+>", " ", cleaned)

        # Decodificar entidades HTML: &nbsp;, &amp;, etc.
        cleaned = unescape(cleaned)
        cleaned = cleaned.replace("\xa0", " ")

        # Normalizar espacios y saltos de línea
        cleaned = re.sub(r"[ \t\r\f\v]+", " ", cleaned)
        cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)
        cleaned = cleaned.strip()

        # Eliminar firmas / bloques automáticos conocidos.
        # Importante: se aplica después de limpiar HTML.
        signature_patterns = [
            r"(?:^|\n)\s*(?:Saludos|Un saludo|Un cordial saludo|Saludos cordiales|Gracias\.?|Atentamente|Cordialmente)\b[\s\S]*$",
            r"(?:^|\n)\s*www\.equipeceramicas\.com\b[\s\S]*$",
            r"(?:^|\n)\s*No es necesario contestar a este correo\b[\s\S]*$",
            r"(?:^|\n)\s*Your request\b[\s\S]*$",
            r"(?:^|\n)\s*Hola,\s*\n+\s*Esto es una respuesta automática\b[\s\S]*$",
        ]

        for pattern in signature_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()

        # Limpieza final
        cleaned = re.sub(r"[ \t\r\f\v]+", " ", cleaned)
        cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)

        return cleaned.strip()
