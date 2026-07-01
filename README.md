# RAGent

Backend en Python que publica agentes conversacionales mediante una API compatible con OpenAI para integrarse con Open WebUI, n8n u otros clientes que consuman `/v1/models` y `/v1/chat/completions`.

RAGent funciona como una capa intermedia que encapsula modelos agénticos. Hacia fuera expone una interfaz estable y OpenAI-compatible; hacia dentro resuelve qué agente ejecutar, qué prompt usar, qué herramientas puede invocar, qué fuentes de conocimiento consultar y contra qué proveedor real de IA debe correr. El cliente no habla directamente con el modelo base configurado: selecciona un agente publicado y RAGent orquesta la ejecución completa.

## API publicada

La API separa tres superficies funcionales:

- Conversación con agentes: endpoints compatibles con OpenAI que ejecutan agentes y pueden usar herramientas y búsqueda internamente.
- Gestión e ingesta de fuentes de conocimiento: endpoints para listar fuentes, crear colecciones y añadir datos indexados.
- Búsqueda directa en fuentes de conocimiento: endpoints de retrieval que devuelven documentos encontrados, sin llamar a un agente ni generar una respuesta conversacional.

## Agentes

Los agentes se definen en `app/core/agent/catalog.py`. Cada definición indica:

- qué identificador público utiliza,
- qué prompt carga,
- si se publica en la API,
- y si puede registrar herramientas.

Actualmente el catálogo incluye:

- `Quipi`: agente público publicado en `/v1/models`.
- `Base`: agente interno sin herramientas.
- `Summarizer`: agente interno usado para tareas auxiliares, como el resumen de tickets durante la ingesta.

El módulo de agentes se reparte en estas piezas principales:

- `AgentCatalog`: lista y resuelve las definiciones de agentes.
- `AgentFactory`: construye agentes ejecutables sobre un backend OpenAI-compatible.
- `AgentService`: coordina historial, ejecución y respuesta compatible con OpenAI.
- `app/core/agent/tools.py`: registra las herramientas disponibles para los agentes habilitados.

Los prompts viven en `app/core/prompts` y se cargan desde `PromptService`. Esto permite publicar distintos comportamientos agénticos sobre el mismo proveedor y el mismo modelo base, desacoplando la integración cliente de la implementación real.

## Arquitectura

Flujo principal de una petición de chat:

1. El cliente llama a `/v1/chat/completions` con `model` igual al identificador del agente.
2. `AgentService` obtiene la definición desde `AgentCatalog`.
3. `AgentFactory` construye el agente con su prompt, su proveedor de chat y sus herramientas.
4. El agente responde directamente o solicita contexto mediante herramientas.
5. Las herramientas consultan Qdrant a través de `KnowledgeSourceRetrievalService` cuando aplica.
6. La API devuelve una respuesta `chat.completion` compatible con OpenAI.

Internamente los agentes se ejecutan con PydanticAI y usan:

- `CHAT_*` para el backend real de chat,
- `EMBEDDING_*` para embeddings,
- `QDRANT_*` para recuperación de contexto.

## Fuentes de conocimiento

Las fuentes se definen en `app/core/knowledge_source/catalog.py`. El catálogo es la referencia para saber qué fuentes existen y cómo se identifican.

La parte de fuentes de conocimiento se organiza así:

- `KnowledgeSourceCatalog`: lista y resuelve las fuentes disponibles.
- `KnowledgeSourceRetrievalService`: recupera contexto relevante para las herramientas de los agentes.
- `KnowledgeSourceService`: coordina las operaciones públicas sobre fuentes de conocimiento.
- `KnowledgeSourceIngestorFactory`: selecciona el servicio de ingesta adecuado para cada fuente.
- `app/core/document_processing`: encapsula el procesado previo a la ingesta, como la conversión de manuales HTML en bloques listos para indexar.

Las fuentes configuradas actualmente son `devices`, `employees`, `manuals` y `tickets`. Todas usan recuperación híbrida sobre Qdrant.

Detalles relevantes:

- `devices`: información de dispositivos, equipos de usuario y servidores.
- `employees`: información de empleados y datos de contacto corporativo.
- `manuals`: manuales HTML convertidos a bloques de texto e imagen para su indexación.
- `tickets`: incidencias de HelpDesk; durante la ingesta se apoya en el agente interno `Summarizer`.

La fuente `manuals` admite la ingesta de ficheros HTML y puede generar embeddings multimodales cuando el modelo de embeddings configurado acepta texto e imagen en el endpoint `/embeddings`.

La API incluye estas rutas principales:

- `GET /knowledge-source`: lista las fuentes disponibles.
- `POST /knowledge-source/{knowledge_source_id}`: crea la colección de una fuente si no existe.
- `POST /knowledge-source/{knowledge_source_id}/points/from-json`: añade o actualiza datos enviados como una lista JSON.
- `POST /knowledge-source/{knowledge_source_id}/points/from-html`: añade o actualiza un manual HTML enviado como fichero.
- `POST /knowledge-source/{knowledge_source_id}/search`: busca documentos directamente en una fuente y devuelve los resultados de retrieval.

Los endpoints de `knowledge-source` devuelven respuestas con `status`, `operation` y `result`. Cuando la operación está asociada a una fuente concreta, también incluyen `knowledge_source_id`. En caso de error devuelven `status="error"` y un objeto `error` con `code`, `message` y, cuando aplica, `details`.

## Configuración

Crea un `.env` a partir de `.env.example` y ajusta los valores a tu entorno real.

Variables principales:

- `CHAT_*`: proveedor real del modelo de chat que ejecutará los agentes.
- `EMBEDDING_*`: proveedor y modelo de embeddings.
- `QDRANT_*`: conexión con Qdrant.
- `LLM_*`: parámetros generales de ejecución.

Puntos importantes:

- `CHAT_BASE_URL` debe apuntar al proveedor real del modelo de chat, por ejemplo un servidor OpenAI-compatible, vLLM o un gateway equivalente.
- `CHAT_BASE_URL` no debe apuntar al propio endpoint `/v1` de RAGent, o entrarías en un bucle de llamadas sobre el mismo servicio.
- Si quieres indexar `manuals` con contexto visual, `EMBEDDING_MODEL` y `EMBEDDING_BASE_URL` deben soportar embeddings multimodales.

## Docker

Hay dos ficheros Compose:

- `docker-compose.local.yml`: levanta `ragent`, `qdrant` y `open-webui` para desarrollo local.
- `docker-compose.yml`: levanta `ragent` y `qdrant` en una red externa `proxy`, pensada para un despliegue integrado con un reverse proxy del host.

Arranque local con Open WebUI:

```bash
docker compose -f docker-compose.local.yml up --build -d
docker compose -f docker-compose.local.yml logs -f ragent
```

Parada:

```bash
docker compose -f docker-compose.local.yml down
```

Puertos por defecto:

- RAGent en `docker-compose.local.yml`: `http://localhost:8000`
- RAGent en `docker-compose.yml`: `http://localhost:8001`
- Qdrant: `http://localhost:6333`
- Open WebUI local: `http://localhost:8080`

Cuando Open WebUI se ejecuta desde `docker-compose.local.yml`, puede configurarse contra RAGent usando la URL interna `http://ragent:8000/v1`. Desde fuera de Docker, la URL equivalente es `http://localhost:8000/v1`.

Si usas `docker-compose.yml`, recuerda que la red externa `proxy` debe existir previamente en el host.

## Evaluaciones

Las evaluaciones se ejecutan en el host local, no dentro del contenedor de `ragent`. El runner en `evals/` llama por HTTP al servicio publicado en `localhost`.

Para ejecutarlas desde cero en otro equipo necesitas:

- tener Docker y Docker Compose disponibles,
- crear un entorno virtual local para el runner,
- instalar `requirements-eval.txt` en ese entorno,
- levantar `ragent` con Docker Compose,
- y ejecutar `python -m evals.run` desde el host.

Ejemplo completo con `docker-compose.local.yml`:

```bash
python -m pip install -r requirements-eval.txt
docker compose -f docker-compose.local.yml up --build -d
python -m evals.run --base-url http://localhost:8000
```

El dataset por defecto es `evals/datasets/quipi_eval.yaml`, pero puedes pasar otro como primer argumento posicional.
