import logging
import random

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from app.api.dependencies import get_qa_service
from app.core.config import settings
from app.models.qa import QARequest, QAResponse
from app.services.background_evaluation import evaluate_in_background
from app.services.qa_service import QAService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/qa",
    response_model=QAResponse,
    status_code=status.HTTP_200_OK,
    summary="Preguntas y respuestas sobre un producto específico",
    description="""
    Responde preguntas específicas sobre un producto del catálogo,
    basándose exclusivamente en su ficha técnica indexada.

    Si se provee `product_id`, se hace lookup directo.
    Si no, se infiere el producto más relevante desde la pregunta.
    """,
)
async def product_qa(
    request: QARequest,
    background_tasks: BackgroundTasks,
    service: QAService = Depends(get_qa_service),
) -> QAResponse:
    logger.info("Q&A request recibido | question: '%s'", request.question[:50])

    try:
        result = service.answer(
            question=request.question,
            product_id=request.product_id,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        error_str = str(e)
        if "503" in error_str or "UNAVAILABLE" in error_str:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="El servicio de IA está temporalmente ocupado. "
                       "Intenta nuevamente en unos segundos.",
            ) from e
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Límite de requests alcanzado. "
                       "Intenta nuevamente en un momento.",
            ) from e
        logger.exception("Error inesperado en Q&A: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno procesando la pregunta. Intenta nuevamente.",
        ) from e

    # Definition of Done. Evaluación online no negociable
    # Mismo patrón que /search: BackgroundTasks + muestreo configurable
    should_evaluate = (
        result.get("trace_id") is not None
        and random.random() < settings.eval_sample_rate
    )

    if should_evaluate:
        background_tasks.add_task(
            evaluate_in_background,
            trace_id=result["trace_id"],
            query=result["question"],
            answer=result["answer"],
            context=result["source_document"],
        )

    return QAResponse(
        question=result["question"],
        answer=result["answer"],
        product_id=result["product_id"],
        product_name=result["product_name"],
        source_document=result["source_document"],
        product_found_via=result["product_found_via"],
    )