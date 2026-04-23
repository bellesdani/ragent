# RAGent

Servicio backend en Python para exponer endpoints compatibles con OpenAI (`/v1/models` y `/v1/chat/completions`) y actuar como una capa de agentes entre Open WebUI y los proveedores reales de IA. Open WebUI ve agentes publicados por este servicio; la resolución del modelo backend, el RAG y las futuras herramientas quedan encapsuladas aquí.

## Estructura

```text
app/
  api/
    routes/
    schemas/
  agent/
  core/
  llm/
  retrieval/
```

## Decisiones técnicas

- `FastAPI` para exponer la API y facilitar compatibilidad con Open WebUI.
- Catálogo de agentes propio: la API publica agentes como `quipi` y `ejemplo`; no expone directamente los modelos backend del proveedor.
- Cliente OpenAI-compatible propio sobre `httpx` para evitar acoplar el núcleo a LangChain y separar proveedor/modelo de chat y embeddings.
- Capa `retrieval` desacoplada mediante interfaz base para poder sustituir Qdrant o la estrategia RAG.
- Búsqueda híbrida ligera: recuperación vectorial en Qdrant y reranking keyword local para no exigir nuevas piezas de indexación.
- Configuración contenida: el `.env` queda reservado a infraestructura y conectividad; identidad del agente, prompt y estrategia de retrieval quedan en código por ahora.

## Modelo Mental

La API no está pensada para reenviar modelos del proveedor a Open WebUI, sino para publicar agentes propios.

- Open WebUI consume `/v1/models` y ve identificadores de agente como `quipi` y `ejemplo`.
- Cuando Open WebUI envía `model=<agent_id>` a `/v1/chat/completions`, este servicio resuelve internamente qué modelo backend usar, qué retrieval aplicar y qué prompt o herramientas asociar.
- `CHAT_MODEL` es interno y pertenece al proveedor configurado.
- `quipi` es el agente RAG principal.
- `ejemplo` es un agente básico de prueba sin RAG para validar conectividad y memoria conversacional.

Con este enfoque, hoy puedes publicar `Quipi` como agente RAG sobre Qdrant y vLLM y `Ejemplo` como smoke test del pipeline, y mañana añadir otros agentes con distinto conocimiento, distinto proveedor o distintas herramientas sin cambiar la integración con Open WebUI.

## Puesta en marcha local

1. Crear entorno e instalar dependencias:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Crear `.env` a partir de `.env.example` y ajustar:

- `QDRANT_URL`
- `CHAT_PROVIDER`
- `CHAT_BASE_URL`
- `CHAT_MODEL`
- `EMBEDDING_PROVIDER`
- `EMBEDDING_BASE_URL`
- `EMBEDDING_MODEL`
- claves/API keys necesarias

Si falta `.env` o no están definidas las variables obligatorias, la aplicación falla al arrancar. La validación se hace en la carga de configuración para evitar que el servicio quede levantado con una configuración inválida.

Por ahora, `Quipi`, `Ejemplo`, sus prompts del sistema y los parámetros de retrieval de Qdrant están definidos en código. El `.env` solo contiene conectividad con proveedores y servicios externos.

3. Lanzar la API:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Despliegue con Docker

Se han dejado dos ficheros `compose` independientes:

- [docker-compose.local.yml](C:/Users/informatica1/Repositorios/ragent/docker-compose.local.yml:1): pensado para desarrollo o despliegue local, con publicación de puertos y `healthcheck`.
- [docker-compose.yml](C:/Users/informatica1/Repositorios/ragent/docker-compose.yml:1): base de despliegue más cercana a producción.

Ambos inyectan configuración mediante `environment:`. No se pasa `.env` al contenedor. Los valores pueden venir del entorno de la máquina o del mecanismo de sustitución de variables de Docker Compose.

1. Definir en tu shell o en el entorno de Docker Compose las variables necesarias.

2. Construir y arrancar en local:

```powershell
docker compose -f docker-compose.local.yml up --build -d
```

3. Ver logs:

```powershell
docker compose -f docker-compose.local.yml logs -f ragent
```

4. Parar el servicio:

```powershell
docker compose -f docker-compose.local.yml down
```

Para producción puedes arrancar directamente con `docker-compose.yml` o crear tu propia variante aparte para proxy, red, labels y políticas específicas.

## Ejemplo de llamada

```bash
curl http://localhost:8000/v1/models
```

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "quipi",
    "messages": [
      {"role": "user", "content": "¿Cuál es la política de vacaciones?"}
    ]
  }'
```

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ejemplo",
    "messages": [
      {"role": "user", "content": "Hola, recuérdame luego que esto es una prueba de conectividad."}
    ]
  }'
```

## Integración con Qdrant

El retriever usa embeddings vía `POST /v1/embeddings` contra el proveedor configurado en `EMBEDDING_BASE_URL` y después consulta Qdrant con `query_points`. La colección, las claves de texto del payload, el límite de búsqueda, el reranking keyword y el recorte de contexto están definidos en código en esta versión.

## Streaming

El endpoint acepta `"stream": true` y responde como `text/event-stream` con chunks SSE en formato OpenAI-compatible básico.
