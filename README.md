# RAGent

Backend en Python que publica agentes conversacionales con una API compatible con OpenAI para integrarse con Open WebUI u otros clientes que usen `/v1/models` y `/v1/chat/completions`.

El servicio no expone directamente el modelo de chat configurado. El cliente selecciona un agente publicado y RAGent se encarga de ejecutarlo contra el proveedor real de IA, con su prompt, herramientas y fuentes de conocimiento cuando corresponda.

## Agentes

Los agentes se definen en `app/core/agent/catalog.py`. Cada definicion indica que prompt usa, si se publica en la API y si puede registrar herramientas.

El modulo de agentes se reparte en estas piezas principales:

- `AgentCatalog`: lista y resuelve las definiciones de agentes.
- `AgentFactory`: construye agentes ejecutables.
- `AgentService`: coordina el historial, la ejecucion y la respuesta compatible con OpenAI.
- `app/core/agent/tools.py`: contiene las herramientas disponibles para los agentes habilitados.

Los prompts viven en `app/core/prompts` y se cargan desde `PromptService`. Pueden existir agentes internos para tareas auxiliares aunque no se publiquen en `/v1/models`.

## Arquitectura

Flujo principal de una peticion de chat:

1. El cliente llama a `/v1/chat/completions` con `model` igual al id del agente.
2. `AgentService` obtiene la definicion desde `AgentCatalog`.
3. `AgentFactory` construye el agente con su prompt y configuracion.
4. El agente responde directamente o solicita contexto mediante herramientas.
5. Las herramientas consultan las fuentes de conocimiento cuando aplica.
6. La API devuelve una respuesta `chat.completion` compatible con OpenAI.

## Fuentes de conocimiento

Las fuentes se definen en `app/core/knowledge_source/catalog.py`. El catalogo es la referencia para saber que fuentes existen y como se identifican.

La parte de fuentes de conocimiento se organiza asi:

- `KnowledgeSourceCatalog`: lista y resuelve las fuentes disponibles.
- `KnowledgeSourceRetrievalService`: recupera contexto relevante para las herramientas de los agentes.
- `KnowledgeSourceService`: coordina las operaciones publicas sobre fuentes de conocimiento.
- `KnowledgeSourceIngestorFactory`: selecciona el servicio de ingesta adecuado cuando una fuente lo necesita.
- `app/core/document_processing`: encapsula el procesado de documentos previo a la ingesta, como la conversion de manuales HTML a bloques de texto e imagen.

Las fuentes configuradas actualmente son `devices`, `employees`, `manuals` y `tickets`. Todas cuentan con un ingestor especifico para crear su coleccion y cargar datos en Qdrant. La fuente `manuals` admite la ingesta de ficheros HTML y puede generar embeddings multimodales cuando el manual incluye imagenes embebidas validas.

La API incluye estas rutas principales:

- `GET /knowledge-source`: lista las fuentes disponibles.
- `POST /knowledge-source/{knowledge_source_id}`: crea la coleccion de una fuente con ingestor.
- `POST /knowledge-source/{knowledge_source_id}/points/from-json`: anade o actualiza datos enviados como JSON.
- `POST /knowledge-source/{knowledge_source_id}/points/from-html`: anade o actualiza un manual HTML enviado como fichero.

Los endpoints de `knowledge-source` devuelven una respuesta con `status`, `operation` y `result`. Cuando la operacion esta asociada a una fuente concreta, tambien incluyen `knowledge_source_id`. En caso de error, devuelven `status="error"` y un objeto `error` con el detalle controlado.

La documentacion interactiva de FastAPI muestra los endpoints y esquemas actuales.

## Configuracion

Crea un `.env` a partir de `.env.example` y ajusta los valores para tu entorno.

- `CHAT_*`: proveedor real del modelo de chat.
- `EMBEDDING_*`: proveedor del modelo de embeddings.
- `QDRANT_*`: conexion con Qdrant.
- `LLM_*`: parametros generales de ejecucion.

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

## Evaluaciones

Las evaluaciones se ejecutan en el host local, no dentro del contenedor de `ragent`. El runner en `evals/` llama por HTTP al servicio publicado en `localhost`.

Para ejecutarlas desde cero en otro equipo necesitas:

- tener Docker y Docker Compose disponibles,
- crear un entorno virtual local para el runner,
- instalar `requirements-dev.txt` en ese entorno,
- levantar `ragent` con Docker Compose,
- y ejecutar `python -m evals.run` desde el host.

Ejemplo completo con `docker-compose.local.yml`:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
docker compose -f docker-compose.local.yml up --build -d
.\.venv\Scripts\python -m evals.run --base-url http://localhost:8000
```

Ejemplo completo con `docker-compose.yml`:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
docker compose -f docker-compose.yml up --build -d
.\.venv\Scripts\python -m evals.run --base-url http://localhost:8001
```

Si ya tienes `.venv` preparada y no necesitas reconstruir la imagen, puedes reutilizar ambas cosas:

```powershell
docker compose -f docker-compose.local.yml up -d
.\.venv\Scripts\python -m evals.run --base-url http://localhost:8000
```
