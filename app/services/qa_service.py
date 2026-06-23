import logging

from app.core.telemetry import TelemetryClient
from app.repositories.vector_store import VectorStoreRepository
from app.services.embedding_service import EmbeddingService
from app.services.gemini_service import GeminiService

logger = logging.getLogger(__name__)

# Prompt diseñado específicamente para Q&A sobre un producto,
# distinto al de búsqueda general. Instrucción explícita de
# admitir falta de información.
_QA_SYSTEM_PROMPT = """Eres un asistente experto en productos electrónicos.
Tu tarea es responder preguntas específicas sobre UN producto concreto,
basándote EXCLUSIVAMENTE en la ficha técnica proporcionada.

Reglas estrictas:
- Responde SOLO con información que esté explícitamente en la ficha técnica
- Si la información solicitada NO está en la ficha, responde claramente:
  "No tengo esa información disponible en la ficha técnica de este producto"
- NUNCA inventes especificaciones, colores, medidas o características
- Sé conciso y directo. Responde la pregunta específica, no hagas un
  resumen completo del producto
- Responde en el mismo idioma de la pregunta del usuario"""


class QAService:
    """
    Orquestador del pipeline Q&A sobre productos específicos.

    Flujo:
    1. Si viene product_id → get_by_id() (lookup directo, sin embedding)
       Si no viene        → embed_text() + search(n_results=1)
    2. Construir prompt con la ficha del producto como contexto
    3. Gemini responde la pregunta basándose SOLO en esa ficha
    4. Devolver respuesta + fuente original (documento indexado)

    La separación respuesta/fuente es la garantía anti-alucinación:
    el LLM genera lenguaje natural, el código extrae los datos duros.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreRepository,
        gemini_service: GeminiService,
        telemetry: TelemetryClient,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._gemini_service = gemini_service
        self._telemetry = telemetry

    def answer(
        self,
        question: str,
        product_id: str | None = None,
    ) -> dict:
        """
        Responde una pregunta sobre un producto específico.

        Args:
            question: pregunta en lenguaje natural del usuario.
            product_id: id exacto del producto. Si es None, se
                infiere desde la pregunta via búsqueda semántica.

        Returns:
            Dict con question, answer, product_id, product_name,
            source_document, product_found_via y trace_id.

        Raises:
            ValueError: si la pregunta está vacía o el producto
                        no existe en el catálogo.
        """
        if not question or not question.strip():
            raise ValueError("La pregunta no puede estar vacía")

        logger.info("Iniciando Q&A | question: '%s'", question[:50])

        lf = self._telemetry.client

        with self._telemetry.observation(
            name="qa_pipeline",
            input={"question": question, "product_id": product_id},
        ):
            # ── PASO 1: Identificar el producto ───────────────────
            if product_id is not None:
                product, found_via = self._get_product_by_id(product_id)
            else:
                product, found_via = self._find_product_from_question(question)

            if product is None:
                raise ValueError(
                    f"No se encontró ningún producto relevante "
                    f"para la pregunta: '{question}'"
                )

            # ── PASO 2: Generar respuesta con Gemini ──────────────
            with self._telemetry.observation(
                name="qa_generate",
                input={
                    "question": question,
                    "product": product["metadata"]["name"],
                },
            ):
                answer = self._generate_answer(
                    question=question,
                    product_document=product["document"],
                )

                if lf:
                    lf.update_current_span(
                        output={"answer": answer},
                        metadata={"found_via": found_via},
                    )

            current_trace_id = None
            if lf:
                current_trace_id = lf.get_current_trace_id()
                lf.update_current_span(
                    output={
                        "answer": answer,
                        "product_id": product["id"],
                        "found_via": found_via,
                    }
                )

        logger.info("Q&A completado | producto: %s", product["metadata"]["name"])

        return {
            "question": question,
            "answer": answer,
            "product_id": product["id"],
            "product_name": product["metadata"]["name"],
            "source_document": product["document"],
            "product_found_via": found_via,
            "trace_id": current_trace_id,
        }

    def _get_product_by_id(self, product_id: str) -> tuple[dict | None, str]:
        """Lookup directo por id (sin embedding, sin búsqueda semántica)."""
        logger.info("Paso 1/2: Lookup directo | product_id: %s", product_id)

        with self._telemetry.observation(
            name="qa_lookup_by_id",
            input={"product_id": product_id},
        ):
            product = self._vector_store.get_by_id(product_id)

        if product is None:
            logger.warning("Producto no encontrado | product_id: %s", product_id)

        return product, "direct_id"

    def _find_product_from_question(
        self, question: str
    ) -> tuple[dict | None, str]:
        """
        Infiere el producto más relevante desde la pregunta
        usando búsqueda semántica con n_results=1.
        """
        logger.info("Paso 1/2: Inferencia semántica del producto")

        with self._telemetry.observation(
            name="qa_embed_question",
            input={"question": question},
        ):
            query_embedding = self._embedding_service.embed_text(question)

        with self._telemetry.observation(
            name="qa_find_product",
            input={"n_results": 1},
        ):
            results = self._vector_store.search(
                query_embedding=query_embedding,
                n_results=1,
            )

        if not results:
            return None, "semantic_search"

        return results[0], "semantic_search"

    def _generate_answer(
        self,
        question: str,
        product_document: str,
    ) -> str:
        """
        Genera la respuesta usando Gemini con el system prompt
        específico de Q&A.

        Usa directamente el cliente de Gemini en lugar de pasar
        por GeminiService.generate_response() porque el prompt
        de sistema es distinto. Q&A necesita respuesta puntual
        con instrucción anti-alucinación, no un ranking de productos.
        """
        from google.genai import types
        from app.core.gemini_retry import call_with_retry

        user_prompt = f"""FICHA TÉCNICA DEL PRODUCTO:
        {product_document}

        PREGUNTA DEL USUARIO:
        {question}

        Responde la pregunta basándote ÚNICAMENTE en la ficha técnica anterior."""

        generated_text = call_with_retry(
            fn=lambda: self._gemini_service._client.models.generate_content(
                model=self._gemini_service._model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_QA_SYSTEM_PROMPT,
                    temperature=0.1,  # más determinístico que /search (0.3)
                    max_output_tokens=512,  # respuesta puntual, no ranking
                ),
            ).text,
            context="qa_generate_answer",
        )

        return generated_text