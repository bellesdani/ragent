import uuid

from datetime import datetime
from app.config import Settings
from qdrant_client import models
from app.core.knowledge_source.ingestion_abc import KnowledgeSourceIngestor
from app.core.document_processing.html_manual_converter import HtmlManualConverter
from app.core.knowledge_source.entities import HtmlManualChunk, HtmlManualDocument, KnowledgeSourceDefinition



class HtmlManualsKnowledgeSourceIngestor(KnowledgeSourceIngestor):
    """
    Este ingestor prepara los manuales HTML para su consulta como fuente de conocimiento. Utiliza:
     - La configuracion base de ingesta (KnowledgeSourceIngestor)
     - El servicio de procesado de manuales HTML (HtmlManualConverter)
     - El modelo de embeddings multimodal configurado

    Funciones publicas:
     - Crear la fuente de conocimiento de manuales (create_knowledge_source).
     - Anadir un manual HTML a la fuente de conocimiento (upsert_knowledge_source_data).
    """

    max_chunk_chars = 2500
    max_image_context_chars = 2500
    max_html_bytes = 10 * 1024 * 1024
    max_image_bytes = 4 * 1024 * 1024

    def __init__(self, settings: Settings, knowledge_source: KnowledgeSourceDefinition) -> None:
        super().__init__(
            settings=settings,
            knowledge_source=knowledge_source,
        )
        self.html_manual_converter = HtmlManualConverter(
            max_html_bytes=self.max_html_bytes,
            max_image_bytes=self.max_image_bytes,
            max_chunk_chars=self.max_chunk_chars,
            max_image_context_chars=self.max_image_context_chars,
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
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.filename",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                lowercase=True, # case-insensitive
                tokenizer=models.TokenizerType.WHITESPACE,
                phrase_matching=True
            )
        )
        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.title",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                lowercase=True, # case-insensitive
                tokenizer=models.TokenizerType.WHITESPACE,
                phrase_matching=True
            )
        )
        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.chunk_type",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
            ),
        )

        return True


    async def upsert_knowledge_source_data(self, data):
        # Primero validamos el modelo de dato que recibimos
        # Esperamos un objeto con filename, content y content_type
        if not isinstance(data, dict):
            raise ValueError("Los manuales HTML deben enviarse como un objeto con filename, content y content_type")
        
        filename = data.get("filename")
        content = data.get("content")
        content_type = data.get("content_type")

        if not isinstance(filename, str):
            raise ValueError("El nombre del fichero HTML no es válido")
        if not isinstance(content, bytes):
            raise ValueError("El contenido del fichero HTML no es válido")
        if content_type is not None and not isinstance(content_type, str):
            raise ValueError("El content_type del fichero HTML no es válido")

        manual_document: HtmlManualDocument = self.html_manual_converter.convert(
            filename=filename,
            content=content,
            content_type=content_type,
        )

        # En segundo lugar, si no existe la colección en Qdrant, la creamos
        if not await self.qdrant_client.collection_exists(self.knowledge_source.collection_name):
            await self.create_knowledge_source()
        # Y eliminamos los puntos del documento si ya existe porque los ids son guids
        #  Si no hacemos esto, tendremos probablemente puntos replicados del mismo documento
        #  Esto, con las otras colecciones no había problema ya que el id del punto, era conocido
        await self.qdrant_client.delete(
            collection_name=self.knowledge_source.collection_name,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key=f"{self.knowledge_source.payload_keys.metadata_key}.filename",
                        match=models.MatchValue(value=filename),
                    )
                ]
            ),
            wait=True,
        )

        # En tercer lugar, preparamos el payload
        # Payload: 
        #  - Semantic_content: Texto para generar los embeddings de búsqueda semántica
        #  - Lexical_content: Texto para procesar con bm25 para la búsqueda léxica  
        #  - Metadata: Información relativa al documento, como bloque, nombre del documento e imágenes
        payloads = []
        for chunk in manual_document.chunks:
            metadata = self._build_metadata(filename, chunk)
            content = chunk.content

            # payload = self._build_payload(manual_document.filename, chunk)
            payload = {
                self.knowledge_source.payload_keys.semantic_content_key: content,
                self.knowledge_source.payload_keys.lexical_content_key: content,
                self.knowledge_source.payload_keys.metadata_key: metadata
            }
            payloads.append(payload)

        # En cuarto lugar, definimos los puntos de Qdrant y antes validamos que hemos configurado los vectores
        # Point:
        #  - id: Identificador del punto
        #  - payload: Payload del punto
        #  - vector: Vectores para la búsqueda, densos para búsqueda semántica y dispersos para busqueda léxica 
        dense_vector_name = self.knowledge_source.dense_vector_name
        sparse_vector_name = self.knowledge_source.sparse_vector_name
        assert dense_vector_name is not None and sparse_vector_name is not None

        points: list[models.PointStruct] = []
        for payload, chunk in zip(payloads, manual_document.chunks):
            points.append(
                models.PointStruct(
                    id=payload[self.knowledge_source.payload_keys.metadata_key]["id"],
                    vector={
                        dense_vector_name: await self.embedding_client.create_multimodal_embedding(
                            input_text=payload[self.knowledge_source.payload_keys.semantic_content_key],
                            image_data_url=chunk.image.data_url if chunk.image else None,
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

        images = [
            chunk.image
            for chunk in manual_document.chunks
            if chunk.image
        ]
        return {
            "filename": manual_document.filename,
            "manuals": 1,
            "chunks": len(manual_document.chunks),
            "images": len(images),
            "embedded_images": len([image for image in images if image.data_url]),
            "points": len(points),
        }


    def _build_metadata(self, filename: str, chunk: HtmlManualChunk):
            metadata = {
                "id": self._build_point_id(filename, chunk),
                "filename": filename,
                "title": chunk.title,
                "chunk_index": chunk.index,
                "chunk_type": chunk.chunk_type,
                "content_length": len(chunk.content),
            }

            if chunk.image:
                metadata["image"] = {
                    "id": chunk.image.id,
                    "mime_type": chunk.image.mime_type,
                    "size_bytes": chunk.image.size_bytes,
                    "alt": chunk.image.alt,
                    "source": chunk.image.source,
                    "embedded": chunk.image.data_url is not None,
                    "skipped_reason": chunk.image.skipped_reason,
                }

            return metadata


    def _build_point_id(self, filename: str, chunk: HtmlManualChunk) -> str:
        identifier = f"{self.knowledge_source.id}:{filename}:{chunk.index}:{chunk.chunk_type}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, identifier))
