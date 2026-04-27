# RAGent

Backend en Python que publica agentes conversacionales con una API compatible con OpenAI para integrarse con Open WebUI u otros clientes que usen `/v1/models` y `/v1/chat/completions`.

El servicio no expone directamente el modelo de chat configurado. El cliente selecciona un agente publico (`Quipi`, `Base`, etc.) y RAGent se encarga de ejecutar ese agente contra el proveedor real de IA, con sus prompts, tools y configuracion.

## Capacidades actuales

- API HTTP con `FastAPI`.
- Endpoints compatibles con OpenAI:
  - `GET /v1/models`
  - `POST /v1/chat/completions`
- Endpoint de salud:
  - `GET /health`
- Runtime de agentes con `PydanticAI`.
- Proveedor de chat OpenAI-compatible configurable por entorno.
- Embeddings OpenAI-compatible para busqueda semantica.
- Recuperacion de contexto en Qdrant.
- Agente con tools de busqueda para empleados, dispositivos y calculadora.

## Agentes publicados

Los agentes disponibles se definen en `app/core/agent_catalog.py`.

### Quipi

Agente corporativo principal.

- Usa el prompt `app/core/prompts/quipi_system.md`.
- Ejecuta el modelo configurado en `CHAT_MODEL`.
- Tiene acceso a tools.
- Puede consultar fuentes de Qdrant sobre empleados y dispositivos.
- Puede usar una calculadora para expresiones aritmeticas simples.

Tools registradas:

- `search_employees(query)`: busca informacion de empleados, contacto, departamento, telefono, extension y correo.
- `search_devices(query)`: busca informacion de equipos, servidores, hosts, hardware, red, sistema operativo, serie, proveedor y modelo.
- `calculator(expression)`: evalua expresiones aritmeticas simples.

### Base

Agente minimo para validar conectividad, historial y generacion de respuestas.

- Usa el prompt `app/core/prompts/base_system.md`.
- Ejecuta el mismo backend de chat que Quipi.
- No registra tools.
- No consulta Qdrant.

## Arquitectura

```text
app/
  main.py                    # crea la aplicacion FastAPI y conecta dependencias
  api/
    routes.py                # endpoints HTTP
    schemas.py               # modelos OpenAI-compatible de request/response
  core/
    agent_catalog.py         # agentes publicados por la API
    agent_factory.py         # construccion del agente PydanticAI y proveedor OpenAI-compatible
    agent_runner.py          # ejecucion del agente, historial y limites de uso
    agent_tools.py           # tools disponibles para los agentes
    chat.py                  # servicio de chat y wiring interno
    config.py                # settings desde .env o entorno
    entities.py              # entidades internas compartidas
    openai.py                # cliente HTTP OpenAI-compatible para embeddings
    prompts.py               # carga cacheada de prompts
    retrieval.py             # busqueda semantica en Qdrant
    prompts/
      base_system.md
      quipi_system.md
```

Flujo principal de una peticion:

1. El cliente llama a `/v1/chat/completions` con `model` igual al id del agente.
2. `ChatAgentService` obtiene la definicion del agente desde `AgentCatalog`.
3. `AgentRunner` separa el ultimo mensaje de usuario y reconstruye el historial para PydanticAI.
4. `AgentFactory` crea un agente con `OpenAIChatModel`, el prompt del agente y sus tools si aplica.
5. El agente decide si responde directamente o si llama a tools.
6. Las tools de busqueda generan embeddings, consultan Qdrant y devuelven contexto al modelo.
7. La API devuelve una respuesta `chat.completion` compatible con OpenAI.

## Fuentes de conocimiento

Las fuentes se definen en `app/core/retrieval.py`.

| Source id | Coleccion Qdrant | Descripcion |
| --- | --- | --- |
| `devices` | `devices` | Dispositivos, servidores, equipos de usuario y planta. |
| `employees` | `employees` | Empleados y datos de contacto corporativo. |

Cada busqueda:

1. Reescribe de forma basica la consulta con los ultimos turnos de conversacion.
2. Genera el embedding con `EMBEDDING_BASE_URL` y `EMBEDDING_MODEL`.
3. Consulta Qdrant con `query_points`.
4. Devuelve hasta `DEFAULT_TOP_K = 15` resultados por fuente.

Los puntos de Qdrant deben contener al menos:

- `payload.content`: texto indexado.
- `payload.metadata`: metadatos devueltos al agente.

## Configuracion

Crea un `.env` a partir de `.env.example` y ajusta los valores. `CHAT_BASE_URL` debe apuntar al proveedor real del modelo de chat, por ejemplo vLLM u otro servidor OpenAI-compatible; no debe apuntar al propio endpoint de RAGent.

```env
CHAT_API_KEY=change-me
CHAT_BASE_URL=http://localhost:8001/v1
CHAT_MODEL=openai/gpt-oss-20b

EMBEDDING_API_KEY=change-me
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-large

LLM_TIMEOUT_SECONDS=60
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=800

QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

Variables obligatorias:

- `CHAT_BASE_URL`
- `CHAT_MODEL`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_MODEL`
- `QDRANT_URL`

Variables opcionales:

- `CHAT_API_KEY`
- `EMBEDDING_API_KEY`
- `QDRANT_API_KEY`
- `LLM_TIMEOUT_SECONDS`
- `LLM_TEMPERATURE`
- `LLM_MAX_TOKENS`

Si falta una variable obligatoria, la aplicacion falla al arrancar con un error de configuracion.

## Arranque local

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Comprueba el servicio:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

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

- RAGent: `http://localhost:8000`
- Open WebUI local: `http://localhost:8080`

Cuando Open WebUI se ejecuta desde `docker-compose.local.yml`, puede configurarse contra RAGent usando la URL interna `http://ragent:8000/v1`. Desde fuera de Docker, la URL equivalente es `http://localhost:8000/v1`.

## Ejemplos de uso

Listar agentes:

```bash
curl http://localhost:8000/v1/models
```

Usar Quipi:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Quipi",
    "messages": [
      {"role": "user", "content": "Que extension tiene el usuario Juan Perez?"}
    ]
  }'
```

Usar Base:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Base",
    "messages": [
      {"role": "user", "content": "Hola, confirma que el servicio responde correctamente."}
    ]
  }'
```

Ejemplo con historial:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Quipi",
    "messages": [
      {"role": "user", "content": "Busca informacion del equipo con hostname PC-001"},
      {"role": "assistant", "content": "He encontrado informacion del equipo PC-001."},
      {"role": "user", "content": "Y quien lo tiene asignado?"}
    ],
    "temperature": 0.2,
    "max_tokens": 800
  }'
```

## Compatibilidad OpenAI

La API implementa el formato basico de chat completions:

- `model`
- `messages`
- `temperature`
- `max_tokens`
- `stream`
- `user`

Validaciones actuales:

- `messages` no puede estar vacio.
- Debe existir al menos un mensaje con `role = "user"`.
- El `model` debe coincidir con un agente publicado.

Limitaciones actuales:

- El streaming SSE aun no esta implementado. Si se envia `"stream": true`, el servicio procesa la peticion como una respuesta no streaming.
- Solo se devuelve una choice por respuesta.
- Los roles de entrada aceptados son `system`, `user`, `assistant` y `tool`, aunque el historial que se pasa al agente se reconstruye con mensajes `system`, `user` y `assistant`.

## Desarrollo

Instalar dependencias:

```powershell
pip install -r requirements.txt
```

Comprobar que los modulos Python compilan:

```powershell
python -m compileall app
```

Dependencias principales:

- `fastapi`
- `uvicorn`
- `httpx`
- `qdrant-client`
- `pydantic-settings`
- `pydantic-ai-slim[openai]`

## Notas de implementacion

- `CHAT_MODEL` es interno al servicio. Open WebUI debe usar los ids de agentes que devuelve `/v1/models`.
- Los prompts viven en ficheros Markdown dentro de `app/core/prompts`.
- La configuracion de agentes y fuentes de Qdrant vive en codigo.
- La calculadora valida caracteres permitidos antes de evaluar la expresion.
- `AgentRunner` limita cada ejecucion a `request_limit=10` llamadas del runtime de PydanticAI.
