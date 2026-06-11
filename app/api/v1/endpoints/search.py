import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_search_service
from app.models.search import ProductResult, SearchRequest, SearchResponse
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)


router = APIRouter()


@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Búsqueda semántica de productos",
    description="""
    Ejecuta el pipeline RAG completo:
    1. Genera embedding semántico de la consulta
    2. Busca productos similares en ChromaDB
    3. Genera respuesta rankeada con justificación via Gemini
    """,
)
async def search_products(
    request: SearchRequest,
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """
    Por qué async:
    FastAPI es async-first. Aunque se manejan servicios síncronos
    internamente (sentence-transformers y ChromaDB),
    declarar el endpoint como async permite que FastAPI maneje
    múltiples requests concurrentes sin bloquear el event loop.

    Por qué Depends():
    FastAPI inyecta automáticamente el SearchService resuelto.
    """
    logger.info("Request recibido | query: '%s'", request.query[:50])

    try:
        result = service.search(
            query=request.query,
            n_results=request.n_results,
            price_max=request.price_max,
            category=request.category,
        )
    except ValueError as e:
        # Error de validación de negocio — 400 Bad Request
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        # Detectar errores transitorios de la API de Gemini
        # para dar al cliente un mensaje accionable
        error_str = str(e)
        if "503" in error_str or "UNAVAILABLE" in error_str:
            logger.warning("Gemini no disponible temporalmente: %s", e)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="El servicio de IA está temporalmente ocupado. "
                       "Intenta nuevamente en unos segundos.",
            ) from e

        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            logger.warning("Límite de requests de Gemini alcanzado: %s", e)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Límite de requests alcanzado. "
                       "Intenta nuevamente en un momento.",
            ) from e

        logger.exception("Error inesperado en búsqueda: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno procesando la búsqueda. Intenta nuevamente.",
        ) from e

    # Mapear el resultado interno al modelo de respuesta Pydantic
    # Este mapeo explícito protege contra exponer campos internos
    # accidentalmente si SearchService cambia su estructura
    try:
        products = [
            ProductResult(
                id=p["id"],
                name=p["metadata"]["name"],
                category=p["metadata"]["category"],
                brand=p["metadata"]["brand"],
                price=p["metadata"]["price"],
                currency=p["metadata"]["currency"],
                similarity_score=p["similarity_score"],
            )
            for p in result["retrieved_products"]
        ]
    except KeyError as e:
        logger.exception("Error mapeando producto — key faltante: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno procesando los resultados.",
        ) from e

    return SearchResponse(
        query=result["query"],
        retrieved_products=products,
        ai_response=result["ai_response"],
        total_found=result["total_found"],
    )