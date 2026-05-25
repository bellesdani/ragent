from app.config import Settings
from qdrant_client import models
from app.core.knowledge_source.ingestion_abc import KnowledgeSourceIngestor
from app.core.knowledge_source.entities import KnowledgeSourceDefinition, Device



class DevicesKnowledgeSourceIngestor(KnowledgeSourceIngestor):
    """
    Este ingestor prepara los dispositivos para su consulta como fuente de conocimiento. Utiliza:
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
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.name",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                lowercase=True, # case-insensitive
                tokenizer=models.TokenizerType.WHITESPACE,
                phrase_matching=False
            ),
        )
        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.type",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                lowercase=True, # case-insensitive
                tokenizer=models.TokenizerType.WHITESPACE,
                phrase_matching=False
            ),
        )
        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.hostname",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
            ),
        )
        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.manufacturer",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                lowercase=True, # case-insensitive
                tokenizer=models.TokenizerType.WHITESPACE,
                phrase_matching=False
            ),
        )
        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.model",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                lowercase=True, # case-insensitive
                tokenizer=models.TokenizerType.WHITESPACE,
                phrase_matching=False
            ),
        )
        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.serial_number",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
            ),
        )
        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.ip_addressess",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
            ),
        )
        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.mac_addresses",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
            ),
        )
        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.location",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                lowercase=True, # case-insensitive
                tokenizer=models.TokenizerType.WHITESPACE,
                phrase_matching=False
            ),
        )
        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.owner",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                lowercase=True, # case-insensitive
                tokenizer=models.TokenizerType.WHITESPACE,
                phrase_matching=False
            ),
        )
        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.user",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
            ),
        )

        return True


    async def upsert_knowledge_source_data(self, data):
        # Primero validamos el modelo de dato que recibimos
        # Esperamos un conjunto de dispositivos
        devices: list[Device] = [
            Device.model_validate(item)
            for item in data
        ]
        
        # En segundo lugar, si no existe la colección en Qdrant, la creamos
        if not await self.qdrant_client.collection_exists(self.knowledge_source.collection_name):
            await self.create_knowledge_source()

        # En tercer lugar, preparamos el payload
        # Payload: 
        #  - semantic_text: Texto con el que generamos los embeddings semánticos para hacer búsqueda semántica
        #  - lexical_text: Texto con el que generamos los embeddings léxicos para hacer búsqueda por caracteres
        #  - device: Json del dispositivo con el que podemos aplicar filtros y ver la información claramente 
        payloads = []
        for device in devices:
            metadata = device.model_dump(mode="json")
            lexical_text = self._build_lexical_text(device)
            semantic_text = self._build_semantical_text(device)

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
            "devices": len(devices),
            "points": len(points),
        }


    def _build_lexical_text(self, device: Device) -> str:
        lines = []
        if device.name:
            lines.append(device.name)
        if device.hostname:
            lines.append(device.hostname)
        if device.type:
            lines.append(device.type)
        if device.operating_system:
            lines.append(device.operating_system)
        if device.architecture:
            lines.append(device.architecture)
        if device.manufacturer:
            lines.append(device.manufacturer)
        if device.model:
            lines.append(device.model)
        if device.serial_number:
            lines.append(device.serial_number)
        if device.owner:
            lines.append(device.owner)
        if device.user:
            lines.append(device.user)
        if device.ip_addresses:
            lines.append(" ".join(map(str, device.ip_addresses)))
        if device.mac_addresses:
            lines.append(" ".join(map(str, device.mac_addresses)))
        if device.vlans:
            lines.append(" ".join(map(str, device.vlans)))
        if device.cpus:
            lines.append(" ".join(map(str, device.cpus)))
        if device.comments:
            lines.append(device.comments)
        return "\n".join(lines)
    

    def _build_semantical_text(self, device: Device):
        lines = []
        if device.name:
            lines.append(f"Dispositivo: {device.name}")
        if device.hostname:
            lines.append(f"Nombre del host: {device.hostname}")
        if device.type:
            lines.append(f"Tipo: {device.type}")
        if device.operating_system:
            lines.append(f"Sistema operativo: {device.operating_system}")
        if device.architecture:
            lines.append(f"Arquitectura: {device.architecture}")
        if device.manufacturer:
            lines.append(f"Fabricante: {device.manufacturer}")
        if device.model:
            lines.append(f"Modelo: {device.model}")
        if device.owner:
            lines.append(f"Propietario: {device.owner}")
        if device.user:
            lines.append(f"Usuario: {device.user}")
        if device.comments:
            lines.append(f"Información adicional: {device.comments}")
        if device.vlans and len(device.vlans) > 0:
            lines.append(f"VLANs: {",".join(map(str, device.vlans))}")
        return ".\n".join(lines)
    
