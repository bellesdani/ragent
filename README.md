# RAGent

Servicio backend en Python para exponer endpoints compatibles con OpenAI (`/v1/models` y `/v1/chat/completions`) y actuar como una capa de agentes entre Open WebUI y los proveedores reales de IA.

## Idea central

Open WebUI no habla directamente con los modelos backend. Habla con agentes publicados por esta API.

- `Quipi`: agente principal con un planner interno que decide si consulta conocimiento corporativo en Qdrant o responde directamente.
- `Base`: agente simple sin RAG, util para validar conectividad, paso de historial y respuestas del modelo.

`CHAT_MODEL` es interno al servicio. El campo `model` que ve Open WebUI representa el agente publico.

## Estructura

```text
app/
  agent/
  api/
    routes/
    schemas/
  config/
  llm/
  retrieval/
```

## Decisiones tecnicas

- `FastAPI` para la API y compatibilidad con Open WebUI.
- Cliente OpenAI-compatible propio sobre `httpx`, sin acoplamiento a LangChain.
- Planner interno para `Quipi`, implementado en codigo propio con salida JSON estructurada.
- Capa `retrieval` desacoplada mediante interfaz base para permitir sustituir la estrategia RAG.
- Configuracion contenida: el `.env` queda reservado a conectividad y credenciales; agentes y parametros de retrieval viven en codigo por ahora.

## Funcionamiento de Quipi

`Quipi` hace una primera llamada al modelo para planificar la respuesta.

- Si el planner decide que no necesita buscar, responde directamente.
- Si el planner decide que si necesita contexto, el backend consulta Qdrant en una o varias fuentes.
- Despues se hace una segunda llamada al modelo con el contexto recuperado para redactar la respuesta final.

El planner no usa `tool calling` nativo del modelo. La decision de buscar o no, y en que fuentes hacerlo, se toma con una salida JSON controlada por el backend. La razon es practica: en este despliegue el modelo puede razonar que debe usar herramientas, pero no siempre materializa `tool_calls` validos en formato OpenAI-compatible.

## Configuracion

Crear `.env` a partir de `.env.example` y ajustar:

- `QDRANT_URL`
- `CHAT_PROVIDER`
- `CHAT_BASE_URL`
- `CHAT_MODEL`
- `EMBEDDING_PROVIDER`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_MODEL`
- claves o API keys necesarias

Si falta una variable obligatoria, la aplicacion falla al arrancar.

## Arranque local

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Docker

Hay dos `compose` independientes:

- [docker-compose.local.yml](C:/Users/informatica1/Repositorios/ragent/docker-compose.local.yml:1): pensado para local, con puertos y `healthcheck`.
- [docker-compose.yml](C:/Users/informatica1/Repositorios/ragent/docker-compose.yml:1): despliegue base.

Ejemplo local:

```powershell
docker compose -f docker-compose.local.yml up --build -d
docker compose -f docker-compose.local.yml logs -f ragent
docker compose -f docker-compose.local.yml down
```

## Ejemplos

Descubrir agentes:

```bash
curl http://localhost:8000/v1/models
```

Usar `Quipi`:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Quipi",
    "messages": [
      {"role": "user", "content": "Cual es la politica de vacaciones?"}
    ]
  }'
```

Usar `Base`:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Base",
    "messages": [
      {"role": "user", "content": "Hola, recuerdame luego que esto es una prueba de conectividad."}
    ]
  }'
```

## Integracion con Qdrant

El retriever usa embeddings via `POST /v1/embeddings` contra el proveedor configurado en `EMBEDDING_BASE_URL` y despues consulta Qdrant con `query_points`.

En esta version, las fuentes de conocimiento disponibles, sus colecciones, las claves de texto del payload y los parametros de retrieval estan definidos en codigo.

## Streaming

El endpoint acepta `"stream": true` y responde como `text/event-stream` con chunks SSE en formato OpenAI-compatible basico.
