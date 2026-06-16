import logging
import time
from typing import Callable

logger = logging.getLogger(__name__)


def call_with_retry(
    fn: Callable[[], str],
    max_retries: int = 5,
    context: str = "",
) -> str:
    """
    Ejecuta una llamada a Gemini con retry y backoff exponencial
    ante rate limit (429 RESOURCE_EXHAUSTED).

    Por qué este helper vive en app/core y no en un servicio
    específico: tanto GeminiService como EvaluationService llaman
    al mismo SDK y enfrentan el mismo error de rate limit. Mantener
    la lógica de retry en un solo lugar evita que ambos servicios
    diverjan con el tiempo (ej: uno con 5 intentos, otro con 3,
    sin razón real para la diferencia).

    Backoff exponencial: 2s, 4s, 8s, 16s, 32s entre intentos.
    Evita reintentar agresivamente, lo cual empeoraría el
    rate limiting en lugar de resolverlo.

    Args:
        fn: función sin argumentos que ejecuta la llamada real
            a Gemini y devuelve el texto de la respuesta.
            Se pasa como callable para que este helper no necesite
            conocer los detalles de cada llamada (prompt, model,
            config) — esos detalles los define quien llama.
        max_retries: intentos máximos antes de propagar el error.
        context: texto descriptivo para logs (ej: "generate_response",
            "evaluate_faithfulness") para facilitar el diagnosticar 
            en qué servicio ocurrió el rate limit.

    Returns:
        El texto de la respuesta de Gemini.

    Raises:
        La excepción original si no es rate limit, o si se
        agotaron los reintentos.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str

            if not is_rate_limit or attempt == max_retries - 1:
                raise

            wait_seconds = 2 ** (attempt + 1)
            logger.warning(
                "Rate limit alcanzado en %s (intento %d/%d). Esperando %ds...",
                context or "llamada a Gemini",
                attempt + 1,
                max_retries,
                wait_seconds,
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Se agotaron los reintentos ante rate limit en {context or 'llamada a Gemini'}"
    )