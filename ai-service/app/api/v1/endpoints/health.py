from fastapi import APIRouter, Depends

from app.api.dependencies import get_vector_store
from app.core.config import settings
from app.models.search import HealthResponse
from app.repositories.vector_store import VectorStoreRepository

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Estado del servicio",
    description="Verifica que la API y sus dependencias están operativas.",
)
async def health_check(
    vector_store: VectorStoreRepository = Depends(get_vector_store),
) -> HealthResponse:
    """
    Por qué un health check es importante en producción:

    Los orquestadores de contenedores (Kubernetes, ECS) hacen
    GET /health cada N segundos. Si responde 200 → instancia sana.
    Si falla → el orquestador reinicia el contenedor automáticamente.

    Para este caso de uso, health check verifica activamente que 
    ChromaDB está accesible y tiene productos indexados.
    """
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        embedding_model=settings.embedding_model,
        llm_model=settings.gemini_model,
        products_indexed=vector_store.count(),
    )