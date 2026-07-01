from fastapi import FastAPI
from app.config import get_settings
from app.core.agent.service import AgentService
from app.api.routers import agent, health, knowledge
from app.core.knowledge_source.service import KnowledgeSourceService


API_DESCRIPTION = (
    "_RAGent_ ofrece una API de chat compatible con OpenAI para agentes,"
    " así como puntos de conexión directos a fuentes de conocimiento para la gestión,"
    " la ingesta y la recuperación de información. \n"
    " - Los puntos de conexión de chat para agentes (*agents*) generan respuestas conversacionales. \n"
    " - Los puntos de conexión a fuentes de conocimiento (*knowlege_sources*) gestionan datos indexados"
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
    return app


app = create_app()
