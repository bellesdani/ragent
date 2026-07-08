from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from app.config import get_settings
from app.core.agent.service import AgentService
from app.api.routers import admin, agent, health, knowledge
from app.core.knowledge_source.service import KnowledgeSourceService


API_DESCRIPTION = (
    "_RAGent_ ofrece una API de chat compatible con OpenAI para agentes,"
    " así como puntos de conexión directos a fuentes de conocimiento para la gestión,"
    " la ingesta y la recuperación de información. \n"
    " - Los puntos de conexión de chat para agentes (*agents*) generan respuestas conversacionales. \n"
    " - Los puntos de conexión a fuentes de conocimiento (*knowledge_sources*) gestionan datos indexados"
    " o devuelven directamente los resultados de la búsqueda."
)


def create_app() -> FastAPI:
    # Cargamos las variables de entorno como settings y los principales servicios
    settings = get_settings()
    knowledge_service = KnowledgeSourceService(settings)
    agent_service = AgentService(settings, knowledge_service)

    # Construimos la app y guardamos los settings y principales servicios como estado de la app
    app = FastAPI(
        title="RAGent",
        description=API_DESCRIPTION,
    )
    app.state.settings = settings
    app.state.agent_service = agent_service
    app.state.knowledge_service = knowledge_service

    # Construimos los routers
    app.include_router(agent.router)
    app.include_router(health.router)
    app.include_router(knowledge.router)
    app.include_router(admin.router)

    # Publicamos el panel de administración como ficheros estáticos en /ui
    app.mount(
        path="/ui",
        app=StaticFiles(directory=Path(__file__).resolve().parent / "static", html=True),
        name="ui",
    )

    @app.get("/", include_in_schema=False)
    async def redirect_to_ui() -> RedirectResponse:
        return RedirectResponse(url="/ui/")

    return app


app = create_app()
