import uuid

from datetime import datetime
from app.config import Settings
from qdrant_client import models
from app.core.knowledge_source.ingestion.abc import KnowledgeSourceIngestor
from app.core.knowledge_source.entities import KnowledgeSourceDefinition, Article



class ArticlesKnowledgeSourceIngestor(KnowledgeSourceIngestor):
    """
    Este ingestor prepara los artículos (PL_ARTÍCULOS) para su consulta como fuente de conocimiento. Utiliza:
     - La configuración base de ingesta (KnowledgeSourceIngestor)

    Funciones públicas:
     - Crear la fuente de conocimiento de dispositivos (create_knowledge_source).
     - Añadir datos de dispositivos a la fuente de conocimiento (upsert_knowledge_source_data).
    """

    def __init__(self, settings: Settings, knowledge_source: KnowledgeSourceDefinition) -> None:
        super().__init__(
            settings=settings,
            knowledge_source=knowledge_source,
        )


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
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.id",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
            ),
        )
        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.adn",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
            ),
        )
        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.sales_reference",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
            ),
        )
        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.ean13",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
            ),
        )
        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.exclusive_customer_id",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
            ),
        )
        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.exclusive_customer_name",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                lowercase=True, # case-insensitive
                tokenizer=models.TokenizerType.WHITESPACE,
                phrase_matching=False
            )
        )
        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.description",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                lowercase=True, # case-insensitive
                tokenizer=models.TokenizerType.WHITESPACE,
                phrase_matching=False
            )
        )

        return True


    async def upsert_knowledge_source_data(self, data):
        # Primero validamos el modelo de dato que recibimos
        # Esperamos un conjunto de artículos (PL_ARTICULOS)
        articles: list[Article] = []
        for item in data:
            try:
                articles.append(Article.model_validate(item))
            except Exception as error:
                print(error)

        # En segundo lugar, si no existe la colección en Qdrant, la creamos
        if not await self.qdrant_client.collection_exists(self.knowledge_source.collection_name):
            await self.create_knowledge_source()

        # En tercer lugar, preparamos el payload
        # Payload: 
        #  - semantic_text: Texto con el que generamos los embeddings semánticos para hacer búsqueda semántica
        #  - lexical_text: Texto con el que generamos los embeddings léxicos para hacer búsqueda por caracteres
        #  - article: Json del artículo con el que podemos aplicar filtros y ver la información claramente 
        payloads = []
        for article in articles:
            metadata = article.model_dump(mode="json")
            lexical_text = self._build_lexical_text(article)
            semantic_text = self._build_semantical_text(article)

            payload = {
                self.knowledge_source.payload_keys.semantic_content_key: semantic_text,
                self.knowledge_source.payload_keys.lexical_content_key: lexical_text,
                self.knowledge_source.payload_keys.metadata_key: metadata
            }
            payloads.append(payload)
                
        # En cuarto lugar, definimos los puntos de Qdrant
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
                    id=self._build_article_id(payload[self.knowledge_source.payload_keys.metadata_key]["id"]),
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

        # Finalmente:
        #  - Insertamos los puntos Qdrant en la colección
        #  - Y actualizamos la metadata de la colección estableciendo last_collection_update
        if len(points) > 0:
            await self.qdrant_client.upsert(
                collection_name=self.knowledge_source.collection_name,
                points=points,
            )
            await self.qdrant_client.update_collection(
                collection_name=self.knowledge_source.collection_name,
                metadata={
                    "last_collection_update": datetime.now().strftime("%Y:%m:%d - %H:%M")
                }
            )

        return {
            "articles": len(articles),
            "points": len(points),
        }


    def _build_lexical_text(self, article: Article) -> str:
        lines = []
        if article.id:
            lines.append(article.id)
        if article.description:
            lines.append(article.description)
        if article.created_by_user:
            lines.append(article.created_by_user)
        if article.adn:
            lines.append(article.adn)
        if article.sales_reference:
            lines.append(article.sales_reference)
        if article.ean13:
            lines.append(article.ean13)
        if article.family_description:
            lines.append(article.family_description)
        if article.subfamily_description:
            lines.append(article.subfamily_description)
        if article.format_description:
            lines.append(article.format_description)
        if article.exclusive_customer_name:
            lines.append(article.exclusive_customer_name)
        return "\n".join(lines)
    

    def _build_semantical_text(self, article: Article) -> str:
        lines = []
        if article.id:
            lines.append(f"Identificador: {article.id}")
        if article.description:
            lines.append(f"Descripción: {article.description}")
        if article.family_id:
            lines.append(f"Familia: {article.family_id} ({article.family_description})")
        if article.subfamily_id:
            lines.append(f"Subfamilia: {article.subfamily_id} ({article.subfamily_description})")
        if article.format_description:
            lines.append(f"Formato: {article.format_description}")
        if article.created_by_user:
            lines.append(f"Creado por el usuario: {article.created_by_user}")
        if article.created_at:
            lines.append(f"Fecha de alta: {article.created_at}")
        if article.deactivated_at:
            lines.append(f"Fecha de baja: {article.deactivated_at}")
        if article.exclusive_customer_name:
            lines.append(f"Cliente exclusivo al que se fabrica: {article.exclusive_customer_name}")
        if article.in_catalog:
            if article.in_catalog == -1:
                lines.append(f"Está en el catálogo: Sí")
            else:
                lines.append(f"Está en el catálogo: No")
        return ".\n".join(lines)
    

    def _build_article_id(self, article_id: str) -> str:
        identifier = f"{self.knowledge_source.id}:{article_id}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, identifier))