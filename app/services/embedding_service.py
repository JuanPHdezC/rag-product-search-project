import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)

# Límite de caracteres antes de que el modelo empiece a truncar.
# all-MiniLM-L6-v2 soporta 256 tokens ≈ 1000 caracteres en español.
# No funciona como un límite duro de la app,es una advertencia temprana
# para detectar textos problemáticos antes de que degraden la calidad.
_MAX_TEXT_LENGTH = 1000


class EmbeddingService:
    """
    Responsabilidad: convertir texto en vectores numéricos.

    Patrón Singleton implícito vía lru_cache: el modelo pesa ~90MB
    y tarda ~3 segundos en cargar. Si se instancia en cada request, 
    la API sería inutilizable. Se instancia una sola vez al
    arrancar la aplicación y se reutiliza en todos los requests.

    En producción, este servicio estaría en un worker dedicado o
    usaría un modelo fine-tuneado en su catálogo específico.
    """

    def __init__(self, model_name: str) -> None:
        logger.info("Cargando modelo de embeddings: %s", model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("Modelo de embeddings cargado correctamente")

    def embed_text(self, text: str) -> list[float]:
        """
        Genera el embedding de un texto individual.

        Args:
            text: Texto a convertir en vector.

        Returns:
            Lista de 384 floats representando el significado del texto.

        Raises:
            ValueError: Si el texto está vacío.

        Por qué convert_to_python=True:
        SentenceTransformer devuelve numpy arrays por defecto.
        ChromaDB espera listas de Python nativas. Esta conversión
        evita errores de tipo silenciosos al indexar o buscar en ChromaDB.
        """
        if not text or not text.strip():
            raise ValueError("El texto para embedding no puede estar vacío")

        # all-MiniLM-L6-v2 trunca silenciosamente textos largos.
        # Advertimos explícitamente para detectar el problema temprano
        # en lugar de obtener embeddings degradados sin saberlo.
        if len(text) > _MAX_TEXT_LENGTH:
            logger.warning(
                "Texto excede %d caracteres (%d). "
                "El modelo truncará el input — el embedding puede degradarse.",
                _MAX_TEXT_LENGTH,
                len(text),
            )

        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Genera embeddings para múltiples textos en una sola llamada.

        Batch es ~10x más eficiente que llamar embed_text() en un loop
        porque el modelo aprovecha paralelismo interno en CPU/GPU.

        Args:
            texts: Lista de textos a embeddear.

        Returns:
            Lista de embeddings en el mismo orden que la entrada.

        Raises:
            ValueError: Si la lista está vacía o algún texto está vacío.
        """
        if not texts:
            raise ValueError("La lista de textos no puede estar vacía")

        # Validar cada texto individualmente para dar un error preciso
        # en lugar de un fallo críptico dentro del modelo
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(
                    f"El texto en índice {i} está vacío. "
                    "Todos los textos deben tener contenido."
                )
            if len(text) > _MAX_TEXT_LENGTH:
                logger.warning(
                    "Texto en índice %d excede %d caracteres (%d). "
                    "El modelo truncará el input.",
                    i,
                    _MAX_TEXT_LENGTH,
                    len(text),
                )

        logger.info("Generando embeddings para %d textos", len(texts))

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32,
        )

        logger.info("Embeddings generados correctamente")
        return embeddings.tolist()


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """
    Factory function con caché
    Garantiza una sola instancia en toda la app.

    @lru_cache(maxsize=1) significa: ejecuta la función la primera vez,
    guarda el resultado, y devuelve el mismo resultado en llamadas
    posteriores sin re-ejecutar. maxsize=1 porque solo necesitamos
    una instancia del servicio.

    """
    return EmbeddingService(model_name=settings.embedding_model)