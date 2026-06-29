import logging

from app.core.telemetry import get_telemetry_client
from app.services.evaluation_service import EvaluationService

logger = logging.getLogger(__name__)

# Instancia única reutilizada entre llamadas en background.
# No usamos lru_cache aquí porque EvaluationService no tiene
# estado costoso de inicializar (a diferencia de EmbeddingService,
# que carga un modelo de 90MB). Instanciar el cliente de Gemini
# es liviano, pero igual evitamos crear uno nuevo por cada
# evaluación en background.
_eval_service = EvaluationService()


def evaluate_in_background(
    trace_id: str,
    query: str,
    answer: str,
    context: str,
) -> None:
    """
    Evalúa Faithfulness y Answer Relevance sobre una respuesta real
    de producción, y envía los scores a LangFuse asociados al trace
    del request original.

    Se ejecuta vía FastAPI BackgroundTasks después de que la
    respuesta ya fue enviada al usuario. Cualquier excepción aquí
    no debe propagarse ni afectar al usuario; el peor caso posible
    es perder esta evaluación puntual, nunca romper el pipeline
    principal.

    Solo evalúa Faithfulness y Answer Relevance (no Context
    Precision/Recall) porque esas dos requieren un golden dataset
    con ground truth, que no existe para tráfico real.

    Args:
        trace_id: ID del trace de LangFuse generado durante el
            request original. Permite asociar este score al
            trace correcto en el dashboard.
        query: consulta original del usuario.
        answer: respuesta generada por Gemini.
        context: texto de los productos recuperados, usado como
            contexto para verificar Faithfulness.
    """
    try:
        faithfulness = _eval_service.evaluate_faithfulness(
            answer=answer,
            context=context,
        )
        relevance = _eval_service.evaluate_answer_relevance(
            query=query,
            answer=answer,
        )

        telemetry = get_telemetry_client()
        if not telemetry.is_enabled:
            logger.debug(
                "LangFuse no habilitado — scores de evaluación online "
                "calculados pero no enviados"
            )
            return

        telemetry.client.create_score(
            trace_id=trace_id,
            name="faithfulness",
            value=faithfulness.score,
        )
        telemetry.client.create_score(
            trace_id=trace_id,
            name="answer_relevance",
            value=relevance.score,
        )
        telemetry.flush()

        logger.info(
            "Evaluación online completada | trace_id=%s | "
            "faithfulness=%.2f | answer_relevance=%.2f",
            trace_id,
            faithfulness.score,
            relevance.score,
        )

    except Exception as e:
        # Nunca propagar. Esto corre después de responder al
        # usuario. Un fallo aquí solo debe perder esta evaluación
        # puntual, registrado en logs, sin ningún otro efecto.
        logger.warning(
            "Error en evaluación online (trace_id=%s): %s. "
            "Esta evaluación puntual se pierde, el pipeline "
            "principal no se ve afectado.",
            trace_id,
            e,
        )