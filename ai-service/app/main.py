import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings

# Configurar logging global de la aplicación
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Factory function para crear la aplicación FastAPI.

    Por qué factory en lugar de instancia global directa:
    Permite crear instancias distintas en tests (con configuración
    diferente) sin afectar la app de producción.
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
## RAG Product Search API

Pipeline de búsqueda semántica de productos usando:
- **sentence-transformers** para embeddings locales
- **ChromaDB** como vector store con persistencia
- **Gemini 2.5 Flash** para generación de respuestas

### Flujo del pipeline
1. La consulta se convierte en un vector semántico
2. ChromaDB encuentra los productos más similares
3. Gemini genera una respuesta rankeada con justificación
        """,
        docs_url="/docs",       # Swagger UI
        redoc_url="/redoc",     # ReDoc (documentación alternativa)
    )

    # CORS: permite que un frontend en otro dominio consuma la API.
    # En producción se reemplaza "*" con los dominios específicos
    # de cada frontend para mayor seguridad.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Montar todos los endpoints bajo /api/v1
    app.include_router(api_router)

    @app.on_event("startup")
    async def startup_event() -> None:
        """
        Se ejecuta una vez cuando arranca el servidor.
        Ideal para pre-cargar recursos pesados (modelos, conexiones DB)
        antes de que llegue el primer request, evitando latencia en
        el primer request real del usuario.
        """
        logger.info("🚀 Arrancando %s v%s", settings.app_name, settings.app_version)

        # Pre-cargar el modelo de embeddings y el cliente Gemini
        # al arrancar en lugar de esperar al primer request
        from app.api.dependencies import get_vector_store
        from app.core.telemetry import get_telemetry_client
        from app.services.embedding_service import get_embedding_service
        from app.services.gemini_service import get_gemini_service

        get_embedding_service()
        get_gemini_service()
        telemetry = get_telemetry_client()
        vector_store = get_vector_store()

        count = vector_store.count()

        if count == 0:
            # WARNING 
            # la app arranca pero informa el problema
            # Un error aquí impediría arrancar incluso en entornos de CI/CD
            # donde se puede tener lógica que poble el vector store después del arranque
            logger.warning(
                "⚠️  Vector store vacío. Los requests de búsqueda fallarán. "
                "Ejecuta: python3 scripts/index_products.py"
            )
        else:
            logger.info("✅ Servicios listos | %d productos indexados", count)
        
        if telemetry.is_enabled:
            logger.info("✅ LangFuse activo | observabilidad habilitada")
        else:
            logger.warning("⚠️  LangFuse no configurado | sin observabilidad")


    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        logger.info("🛑 Apagando %s", settings.app_name)

        # Flush garantiza que ningún trace pendiente se pierda
        # durante el shutdown de la aplicación
        from app.core.telemetry import get_telemetry_client
        get_telemetry_client().flush()

    return app


# Instancia global que uvicorn importa para servir
app = create_app()