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

    def __init__(self, settings: Settings, agent_service: AgentService, definition: KnowledgeSourceDefinition) -> None:
        super().__init__(
            settings=settings,
            definition=definition,
            agent_service=agent_service,
        )


    async def create_knowledge_source(self):
        if await self.qdrant_client.collection_exists(self.knowledge_source.collection_name):
            return False

        if self.knowledge_source.dense_vector_name is None:
            raise ValueError(f"La fuente de conocimiento '{self.knowledge_source.id}' no tiene vector denso configurado.")
        if self.knowledge_source.sparse_vector_name is None:
            raise ValueError(f"La fuente de conocimiento '{self.knowledge_source.id}' no tiene vector disperso configurado.")
                 
        await self.qdrant_client.create_collection(
            collection_name=self.knowledge_source.collection_name,
            vectors_config={
                self.knowledge_source.dense_vector_name: models.VectorParams(
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
                self.knowledge_source.sparse_vector_name: models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=True,
                    ),
                    modifier=models.Modifier.IDF,
                )
            },
        )

        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name="metadata.ticket_title",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                lowercase=True,
                tokenizer=models.TokenizerType.WHITESPACE,
                phrase_matching=True
            )
        )

        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name="metadata.ticket_number",
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
                    ticket_id = ticket_article.ticket_id,
                    ticket_group_id = ticket_article.ticket_group_id,
                    ticket_priority_id = ticket_article.ticket_priority_id,
                    ticket_state_id = ticket_article.ticket_state_id,
                    ticket_organization_id = ticket_article.ticket_organization_id,
                    ticket_number = ticket_article.ticket_number,
                    ticket_title = ticket_article.ticket_title,
                    ticket_created_at = ticket_article.ticket_created_at,
                    ticket_closed_at = ticket_article.ticket_closed_at,
                    ticket_customer_firstname = ticket_article.ticket_customer_firstname,
                    ticket_customer_lastname = ticket_article.ticket_customer_lastname,
                    ticket_customer_department = ticket_article.ticket_customer_department,
                    ticket_customer_email = ticket_article.ticket_customer_email,
                    ticket_creator_firstname = ticket_article.ticket_creator_firstname,
                    ticket_creator_lastname = ticket_article.ticket_creator_lastname,
                    ticket_creator_department = ticket_article.ticket_creator_department,
                    ticket_creator_email = ticket_article.ticket_creator_email,
                    articles = [
                        TicketArticle(
                            article_id = ticket_article.article_id,
                            article_from = ticket_article.article_from,
                            article_to = ticket_article.article_to,
                            article_subject = ticket_article.article_subject,
                            article_content_type = ticket_article.article_content_type,
                            article_body = ticket_article.article_body,
                            article_internal = ticket_article.article_internal,
                            article_created_at = ticket_article.article_created_at,
                            article_creator_firstname = ticket_article.article_creator_firstname,
                            article_creator_lastname = ticket_article.article_creator_lastname,
                            article_creator_department = ticket_article.article_creator_department,
                            article_creator_email = ticket_article.article_creator_email,
                        )
                    ]
                )
            
            # Si ya existe, solo añadimos el artículo
            else:
                tickets[ticket_article.ticket_id].articles.append(
                    TicketArticle(
                        article_id = ticket_article.article_id,
                        article_from = ticket_article.article_from,
                        article_to = ticket_article.article_to,
                        article_subject = ticket_article.article_subject,
                        article_content_type = ticket_article.article_content_type,
                        article_body = ticket_article.article_body,
                        article_internal = ticket_article.article_internal,
                        article_created_at = ticket_article.article_created_at,
                        article_creator_firstname = ticket_article.article_creator_firstname,
                        article_creator_lastname = ticket_article.article_creator_lastname,
                        article_creator_department = ticket_article.article_creator_department,
                        article_creator_email = ticket_article.article_creator_email,
                    )
                )

        # En cuarto lugar, preparamos el payload
        # Payload: 
        #  - Content: Text to generate embeddings and item to be used for semantic search  
        #  - Metadata: Text to use for search and payload metadata
        payloads = []
        for ticket in tickets.values():
            content = self._build_ticket_content(ticket)
            summarized_content = await self._build_ticket_summarize_content(content)
            metadata = self._build_ticket_metadata(ticket)

            payload = {
                "content": content,
                "summarized_content": summarized_content,
                "metadata": metadata
            }
            payloads.append(payload)
                

        # En quinto lugar, definimos los puntos de Qdrant
        # Point:
        #  - id: Identificador del punto
        #  - payload: Payload del punto
        #  - vector: Vectores para la búsqueda, densos para búsqueda semántica y dispersos para busqueda léxica 
        points: list[models.PointStruct] = []
        for payload in payloads:
            # Calculamos el embedding
            embedding = await self.embedding_client.create_embedding(
                input_text=payload["summarized_content"],
                model=self.settings.embedding_model
            )
            # Preparamos el embedding
            points.append(
                models.PointStruct(
                    id=payload["metadata"]["ticket_id"],
                    vector={
                        self.knowledge_source.dense_vector_name: embedding,
                        self.knowledge_source.sparse_vector_name: models.Document(
                            text=payload["content"],
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


    def _build_ticket_content(self, ticket: Ticket) -> str:
        lines :list[str] = []
        lines.append(f"Ticket: {ticket.ticket_title}")
        for article in ticket.articles:
                article_body_cleaned = self._clean_text(article.article_body)
                if article.article_from:
                    lines.append(f" - Mensaje de: {article.article_from}")
                if article.article_to:
                    lines.append(f" - Para: {article.article_to}")
                if article.article_subject:
                    lines.append(f" - Tema: {article.article_subject}")
                if article_body_cleaned:
                    lines.append(f" - Contenido: {article_body_cleaned}")
                lines.append("")
        return "\n".join(lines)    
    

    async def _build_ticket_summarize_content(self, content: str):
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
    

    async def _get_embedding_size(self) -> int:
        embedding = await self.embedding_client.create_embedding(
            input_text="ticket",
            model=self.settings.embedding_model,
        )
        return len(embedding)
    

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
