import logging
from functools import lru_cache

from google import genai
from google.genai import types

from app.core.config import settings
from app.core.gemini_retry import call_with_retry

logger = logging.getLogger(__name__)


class GeminiService:
    """
    Responsabilidad: Gestionar la comunicación con la API de Gemini.

    Esta capa de abstracción es importante por dos razones:

    1. Si Google cambia su API o migramos a otro LLM (OpenAI, Anthropic),
       solo cambia este archivo. El resto del pipeline no se toca.

    2. Centraliza la configuración del modelo: temperatura, tokens,
       system prompt.

    En producción, esta capa también manejaría:
    - Rate limiting y retry logic
    - Logging de costos por request
    - A/B testing entre modelos
    """

    # System prompt define el ROL y COMPORTAMIENTO del modelo.
    # Va separado del user prompt por una razón técnica:
    # el system prompt se cachea en el lado de Google,
    # reduciendo latencia y costos en requests repetidos.
    _SYSTEM_PROMPT = """Eres un asistente experto en recomendación de productos 
    electrónicos. Tu rol es analizar productos recuperados de un catálogo y 
    presentarlos de forma clara, honesta y útil al usuario.

    Reglas que debes seguir siempre:
    - Recomienda SOLO productos del catálogo proporcionado, nunca inventes productos
    - Sé específico sobre por qué cada producto es relevante para la consulta
    - Menciona precio, características clave y para quién es ideal cada producto
    - Si ningún producto es realmente relevante para la consulta, dilo honestamente
    - Responde siempre en el mismo idioma de la consulta del usuario
    - Sé conciso: máximo 3-4 líneas por producto recomendado"""

    def __init__(self) -> None:
        logger.info("Inicializando cliente Gemini con modelo: %s", settings.gemini_model)

        # Inicializar el cliente con la API key
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model

        logger.info("Cliente Gemini inicializado correctamente")

    def generate_response(
        self,
        user_query: str,
        retrieved_products: list[dict],
    ) -> str:
        """
        Genera una respuesta en lenguaje natural basada en los productos
        recuperados por ChromaDB. Este es el corazón del patrón RAG.

        El flujo RAG completo ocurre aquí:
        - RETRIEVAL (ya ocurrió): tenemos los productos más similares
        - AUGMENTATION: inyectamos esos productos como contexto al LLM
        - GENERATION: Gemini genera respuesta basada en ese contexto real



        Args:

            user_query: La consulta original del usuario sin modificar.

            retrieved_products: Lista de productos devueltos por ChromaDB,
                                cada uno con document, metadata y similarity_score.



        Returns:

            Respuesta en lenguaje natural con recomendaciones justificadas.

        Raises:
            ValueError: Si la consulta está vacía o los productos
                        tienen estructura inválida.
        """
        if not user_query or not user_query.strip():
            raise ValueError("La consulta del usuario no puede estar vacía")

        if not retrieved_products:
            return "No encontré productos relevantes para tu consulta en el catálogo."

        # Validar que cada producto tiene la estructura que este método necesita.
        # Aunque los datos vienen de ChromaDB (fuente interna), validamos
        # defensivamente: si VectorStoreRepository cambia su estructura
        # de respuesta, se captura el error sin propagarlo.
        required_keys = {"metadata", "document", "similarity_score"}
        for i, product in enumerate(retrieved_products):
            missing = required_keys - product.keys()
            if missing:
                raise ValueError(
                    f"Producto en índice {i} tiene estructura inválida. "
                    f"Keys faltantes: {missing}"
                )

        products_context = self._build_products_context(retrieved_products)

        user_prompt = f"""CONSULTA DEL USUARIO:
        {user_query}

        PRODUCTOS DISPONIBLES EN EL CATÁLOGO:
        {products_context}

        Basándote ÚNICAMENTE en los productos listados arriba, recomienda los más 
        relevantes para la consulta. Explica brevemente por qué cada uno es una 
        buena opción y menciona su precio."""

        logger.info("Enviando request a Gemini | query: '%s'", user_query[:50])

        generated_text = call_with_retry(
            fn=lambda: self._client.models.generate_content(
                model=self._model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self._SYSTEM_PROMPT,
                    temperature=0.3,
                    max_output_tokens=1024,
                ),
            ).text,
            context="generate_response",
        )

        logger.info("Respuesta generada: %d caracteres", len(generated_text))

        return generated_text

    def _build_products_context(self, products: list[dict]) -> str:
        """
        Convierte la lista de productos recuperados en un texto
        estructurado que Gemini puede leer y razonar fácilmente.

        Por qué estructuramos el contexto así:
        Los LLMs procesan mejor información cuando está claramente
        delimitada y etiquetada. Sin estructura, el modelo puede
        confundir datos entre productos.

        Incluimos similarity_score para que Gemini sepa qué tan
        relevante encontró ChromaDB cada producto, aprovechando esa
        información para priorizar su respuesta.
        """
        context_parts = []

        for i, product in enumerate(products, start=1):
            metadata = product["metadata"]
            score = product["similarity_score"]

            product_text = (
                f"PRODUCTO {i} (relevancia: {score}):\n"
                f"  Nombre: {metadata['name']}\n"
                f"  Categoría: {metadata['category']}\n"
                f"  Marca: {metadata['brand']}\n"
                f"  Precio: {metadata['price']} {metadata['currency']}\n"
                f"  Descripción: {product['document']}\n"
            )
            context_parts.append(product_text)

        return "\n".join(context_parts)


@lru_cache(maxsize=1)
def get_gemini_service() -> GeminiService:
    """
    Factory function con caché.
    El cliente HTTP de Gemini se inicializa una vez y se reutiliza.
    """
    return GeminiService()