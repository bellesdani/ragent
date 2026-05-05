# RAGent

Backend en Python que publica agentes conversacionales con una API compatible con OpenAI para integrarse con Open WebUI u otros clientes que usen `/v1/models` y `/v1/chat/completions`.

El servicio no expone directamente el modelo de chat configurado. El cliente selecciona un agente publico (`Quipi`, `Base`, etc.) y RAGent se encarga de ejecutar ese agente contra el proveedor real de IA, con sus prompts, tools y configuracion.

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

Flujo principal de una peticion:

1. El cliente llama a `/v1/chat/completions` con `model` igual al id del agente.
2. `ChatAgentService` obtiene la definicion del agente desde `AgentCatalog`.
3. `AgentRunner` separa el ultimo mensaje de usuario y reconstruye el historial para PydanticAI.
4. `AgentFactory` crea un agente con `OpenAIChatModel`, el prompt del agente y sus tools si aplica.
5. El agente decide si responde directamente o si llama a tools.
6. Las tools de busqueda generan embeddings, consultan Qdrant y devuelven contexto al modelo.
7. La API devuelve una respuesta `chat.completion` compatible con OpenAI.

## Fuentes de conocimiento

Las fuentes se definen en `app/core/qdrant_collections.py` y la busqueda se ejecuta desde `app/core/qdrant_retrieval.py`.

| Source id | Coleccion Qdrant | Descripcion |
| --------- | ---------------- | ----------- |
| `devices` | `devices` | Dispositivos, servidores, equipos de usuario y planta. |
| `employees` | `employees` | Empleados y datos de contacto corporativo. |
| `manuals` | `manuals` | Manuales de software y operativas habituales. |
| `tickets` | `tickets` | Tickets registrados en Helpdesk. |

Cada búsqueda:

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
QDRANT_API_KEY=change-me
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
