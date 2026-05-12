# RAGent

Backend en Python que publica agentes conversacionales con una API compatible con OpenAI para integrarse con Open WebUI u otros clientes que usen `/v1/models` y `/v1/chat/completions`.

El servicio no expone directamente el modelo de chat configurado. El cliente selecciona un agente publicado y RAGent se encarga de ejecutarlo contra el proveedor real de IA, con su prompt, herramientas y fuentes de conocimiento cuando corresponda.

## Agentes

Los agentes se definen en `app/core/agent/catalog.py`. Cada definición indica qué prompt usa, si se publica en la API y si puede registrar herramientas.

El módulo de agentes se reparte en estas piezas principales:

- `AgentCatalog`: lista y resuelve las definiciones de agentes.
- `AgentFactory`: construye agentes ejecutables.
- `AgentService`: coordina el historial, la ejecución y la respuesta compatible con OpenAI.
- `app/core/agent/tools.py`: contiene las herramientas disponibles para los agentes habilitados.

Los prompts viven en `app/core/prompts` y se cargan desde `PromptService`. Pueden existir agentes internos para tareas auxiliares aunque no se publiquen en `/v1/models`.

## Arquitectura

Flujo principal de una petición de chat:

1. El cliente llama a `/v1/chat/completions` con `model` igual al id del agente.
2. `AgentService` obtiene la definición desde `AgentCatalog`.
3. `AgentFactory` construye el agente con su prompt y configuración.
4. El agente responde directamente o solicita contexto mediante herramientas.
5. Las herramientas consultan las fuentes de conocimiento cuando aplica.
6. La API devuelve una respuesta `chat.completion` compatible con OpenAI.

## Fuentes de conocimiento

Las fuentes se definen en `app/core/knowledge_source/catalog.py`. El catálogo es la referencia para saber qué fuentes existen y cómo se identifican.

La parte de fuentes de conocimiento se organiza así:

- `KnowledgeSourceCatalog`: lista y resuelve las fuentes disponibles.
- `KnowledgeSourceRetrievalService`: recupera contexto relevante para las herramientas de los agentes.
- `KnowledgeSourceService`: coordina las operaciones públicas sobre fuentes de conocimiento.
- `KnowledgeSourceIngestorFactory`: selecciona el servicio de ingesta adecuado cuando una fuente lo necesita.

Las fuentes configuradas actualmente son `devices`, `employees`, `manuals` y `tickets`. Las fuentes `devices`, `employees` y `tickets` tienen ingestor para crear la colección y añadir datos; `manuals` está disponible como fuente semántica si la colección ya existe.

La API incluye estas rutas principales:

- `GET /knowledge-source`: lista las fuentes disponibles.
- `POST /knowledge-source/{knowledge_source_id}`: crea la colección de una fuente con ingestor.
- `POST /knowledge-source/{knowledge_source_id}/points`: añade o actualiza datos de una fuente con ingestor.

La documentación interactiva de FastAPI muestra los endpoints y esquemas actuales.

## Configuración

Crea un `.env` a partir de `.env.example` y ajusta los valores para tu entorno.

- `CHAT_*`: proveedor real del modelo de chat.
- `EMBEDDING_*`: proveedor del modelo de embeddings.
- `QDRANT_*`: conexión con Qdrant.
- `LLM_*`: parámetros generales de ejecución.

`CHAT_BASE_URL` debe apuntar al proveedor real del modelo de chat, por ejemplo vLLM u otro servidor OpenAI-compatible; no debe apuntar al propio endpoint de RAGent.

## Docker

Hay dos ficheros Compose:

- `docker-compose.yml`: levanta solo `ragent`.
- `docker-compose.local.yml`: levanta `ragent` y `open-webui` para desarrollo local.

Arranque local con Open WebUI:

```powershell
docker compose -f docker-compose.local.yml up --build -d
docker compose -f docker-compose.local.yml logs -f ragent
```

Parada:

```powershell
docker compose -f docker-compose.local.yml down
```

Puertos por defecto:

- RAGent en `docker-compose.local.yml`: `http://localhost:8000`
- RAGent en `docker-compose.yml`: `http://localhost:8001`
- Open WebUI local: `http://localhost:8080`

Cuando Open WebUI se ejecuta desde `docker-compose.local.yml`, puede configurarse contra RAGent usando la URL interna `http://ragent:8000/v1`. Desde fuera de Docker, la URL equivalente es `http://localhost:8000/v1`.
