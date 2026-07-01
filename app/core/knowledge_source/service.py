from app.config import Settings
from qdrant_client.models import Filter
from qdrant_client import AsyncQdrantClient
from app.core.utils.embeddings import EmbeddingService
from app.core.knowledge_source.catalog import KnowledgeSourceCatalog
from app.core.knowledge_source.factory_ingestor import KnowledgeSourceIngestorFactory
from app.core.knowledge_source.factory_retrieval import KnowledgeSourceRetrievalFactory
from app.core.knowledge_source.entities import KnowledgeSourceDefinition, RetrievedContext


class KnowledgeSourceService:
    """
    Este servicio es la fachada de fuentes de conocimiento. Coordina el catálogo,
    la ingesta y la recuperación directa de documentos.

    Funciones públicas:
     - Listar las fuentes de conocimiento disponibles (list_knowledge_sources).
     - Crear una nueva fuente de conocimiento (create_knowledge_source).
     - Añadir datos a una fuente de conocimiento (upsert_knowledge_source_data).
     - Buscar documentos en una fuente concreta desde la API (search_knowledge_source).
     - Recuperar documentos relevantes para una consulta (retrieve).
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.catalog = KnowledgeSourceCatalog()
        self.embedding_client = EmbeddingService(
            settings=settings,
        )
        self.qdrant_client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
        self.ingestor_factory = KnowledgeSourceIngestorFactory(
            settings=settings,
        )
        self.retrieval_factory = KnowledgeSourceRetrievalFactory(
            settings=settings,
            embedding_client=self.embedding_client,
            qdrant_client=self.qdrant_client,
        )


    def list_knowledge_sources(self) -> list[KnowledgeSourceDefinition]:
        return self.catalog.list_knowledge_sources()


    async def create_knowledge_source(self, knowledge_source_id: str):
        definition = self.catalog.get_knowledge_source(knowledge_source_id)
        ingestor = self.ingestor_factory.build(definition)
        return await ingestor.create_knowledge_source()


    async def upsert_knowledge_source_data(self, knowledge_source_id: str, data):
        definition = self.catalog.get_knowledge_source(knowledge_source_id)
        ingestor = self.ingestor_factory.build(definition)
        return await ingestor.upsert_knowledge_source_data(data)


    async def search_knowledge_source(self, knowledge_source_id: str, query: str, limit: int) -> RetrievedContext:
        return await self.retrieve(
            query=query,
            limit=limit,
            source_id=knowledge_source_id,
        )


    async def retrieve(self, query: str, limit: int, source_id: str, query_filter: Filter | None = None) -> RetrievedContext:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("La query de busqueda no puede estar vacia")

        source = self.catalog.get_knowledge_source(source_id)
        retriever = self.retrieval_factory.build(source)

        last_collection_update = None
        collection_info = await self.qdrant_client.get_collection(collection_name=source.collection_name)
        if collection_info.config.metadata:
            last_collection_update = collection_info.config.metadata.get("last_collection_update")

        documents = await retriever.retrieve(
            query=cleaned_query,
            source=source,
            limit=limit,
            query_filter=query_filter,
        )

        return RetrievedContext(
            query=cleaned_query,
            documents=documents,
            last_data_update=last_collection_update,
        )
