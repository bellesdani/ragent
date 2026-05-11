from app.config import Settings
from qdrant_client import models
from app.core.knowledge_source.ingestion_abc import KnowledgeSourceIngestor
from app.core.knowledge_source.entities import KnowledgeSourceDefinition, Employee



class EmployeesKnowledgeSourceIngestor(KnowledgeSourceIngestor):
    """
    Este ingestor prepara los empleados para su consulta como fuente de conocimiento. Utiliza:
     - La configuración base de ingesta (KnowledgeSourceIngestor)

    Funciones públicas:
     - Crear la fuente de conocimiento de tickets (create_knowledge_source).
     - Añadir datos de tickets a la fuente de conocimiento (upsert_knowledge_source_data).
    """

    def __init__(self, settings: Settings, knowledge_source: KnowledgeSourceDefinition) -> None:
        super().__init__(
            settings=settings,
            knowledge_source=knowledge_source,
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
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.phone_numbers",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
            ),
        )

        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.phone_extensions",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
            ),
        )


        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.usernames",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
            ),
        )


        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.emails",
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD,
            ),
        )

        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.department",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                lowercase=True,
                tokenizer=models.TokenizerType.WHITESPACE,
                phrase_matching=True
            )
        )

        await self.qdrant_client.create_payload_index(
            collection_name=self.knowledge_source.collection_name,
            field_name=f"{self.knowledge_source.payload_keys.metadata_key}.full_name",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                lowercase=True,
                tokenizer=models.TokenizerType.WHITESPACE,
                phrase_matching=True
            )
        )

        return True


    async def upsert_knowledge_source_data(self, data):
        # Primero validamos el modelo de dato que recibimos
        # Esperamos un conjunto de empleados
        employees: list[Employee] = [
            Employee.model_validate(item)
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
        for employee in employees:
            metadata = self._build_metadata(employee)
            lexical_text = self._build_lexical_text(employee)
            semantic_text = self._build_semantical_text(employee)

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
        points: list[models.PointStruct] = []
        for payload in payloads:
            points.append(
                models.PointStruct(
                    id=payload[self.knowledge_source.payload_keys.metadata_key]["id"],
                    vector={
                        self.knowledge_source.dense_vector_name: await self.embedding_client.create_embedding(
                            input_text=payload[self.knowledge_source.payload_keys.semantic_content_key],
                            model=self.settings.embedding_model
                        ),
                        self.knowledge_source.sparse_vector_name: models.Document(
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
            "employees": len(employees),
            "points": len(points),
        }


    def _build_metadata(self, employee: Employee):
        employee_dict = employee.model_dump(mode="json")
        employee_dict['usernames'] = [ email.split('@')[0] for email in employee.emails]
        employee_dict['phone_numbers'] = [ phone.number for phone in employee.phones]
        employee_dict['phone_extensions'] = [ phone.extension for phone in employee.phones]
        return employee_dict


    def _build_lexical_text(self, employee: Employee) -> str:
        lines = []
        if employee.full_name and employee.alias:
            lines.append(f"Nombre: {employee.full_name} ({employee.alias})")
        if employee.full_name and not employee.alias:
            lines.append(f"Nombre: {employee.full_name}")
        if employee.department:
            lines.append(f"Departamento: {employee.department}")
        if employee.emails:
            lines.append(f"Emails: {",".join(map(str, employee.emails))}")
        if employee.phones:
            phones_str = []
            for phone in employee.phones:
                if phone.extension:
                    phones_str.append(f"{phone.number} (Extensión: {phone.extension})")
                else:
                    phones_str.append(f"{phone.number}")
            lines.append(f"Teléfonos: {",".join(phones_str)}")
        return "\n".join(lines)
    

    def _build_semantical_text(self, employee: Employee):
        lines = []
        if employee.full_name and employee.alias:
            lines.append(f"Nombre: {employee.full_name} ({employee.alias})")
        if employee.full_name and not employee.alias:
            lines.append(f"Nombre: {employee.full_name}")
        if employee.department:
            lines.append(f"Departamento: {employee.department}")
        if employee.emails:
            lines.append(f"Emails: {",".join(map(str, employee.emails))}")
        if employee.phones:
            phones_str = []
            for phone in employee.phones:
                if phone.extension:
                    phones_str.append(f"{phone.number} (Extensión: {phone.extension})")
                else:
                    phones_str.append(f"{phone.number}")
            lines.append(f"Teléfonos: {",".join(phones_str)}")
        return "\n".join(lines)
    
