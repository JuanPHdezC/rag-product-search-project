import logging

from app.repositories.vector_store import VectorStoreRepository
from app.services.embedding_service import EmbeddingService
from app.services.gemini_service import GeminiService

logger = logging.getLogger(__name__)


class SearchService:
    """
    Orquestador del pipeline RAG completo.

    Responsabilidad: coordinar el flujo entre los tres componentes 
    principales del pipeline RAG.

    Principio de inversión de dependencias (DIP):
    SearchService depende de abstracciones (las interfaces de cada
    servicio), no de implementaciones concretas. Las dependencias
    se inyectan desde afuera (Dependency Injection).

    Beneficio práctico: para testear SearchService podemos inyectar
    mocks de EmbeddingService, VectorStore y GeminiService sin
    tocar APIs externas ni cargar modelos reales.

    Así se ve el pipeline completo que orquesta este servicio:

    consulta (str)
         ↓
    embed_text()          → vector [384 floats]
         ↓
    vector_store.search() → top-k productos similares
         ↓
    gemini.generate()     → respuesta en lenguaje natural
         ↓
    SearchResult          → respuesta estructurada al cliente
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreRepository,
        gemini_service: GeminiService,
    ) -> None:
        # Recibe dependencias inyectadas, no las crea internamente
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._gemini_service = gemini_service

    def search(
        self,
        query: str,
        n_results: int = 5,
        price_max: float | None = None,
        category: str | None = None,
    ) -> dict:
        """
        Ejecuta el pipeline RAG completo para una consulta.

        Args:
            query: Consulta en lenguaje natural del usuario.
            n_results: Cuántos productos recuperar de ChromaDB.
            price_max: Filtro opcional de precio máximo.
            category: Filtro opcional por categoría de producto.



        Returns:
            Dict con:
            - query: consulta original
            - retrieved_products: productos encontrados por ChromaDB
            - ai_response: respuesta generada por Gemini
            - total_found: cantidad de productos recuperados

        Raises:
            ValueError: Si los parámetros de entrada son inválidos.
        """
        # Validación defensiva en la capa de servicio.
        # Se complementa la validación via Pydantic en el endpoint,
        #buscando garantizar el contrato independientemente de 
        # quién llame al servicio (SearchService puede ser llamado 
        # desde scripts o tests sin pasar por la API)
        if not query or not query.strip():
            raise ValueError("La consulta no puede estar vacía")

        if not 1 <= n_results <= 20:
            raise ValueError("n_results debe estar entre 1 y 20")

        if price_max is not None and price_max <= 0:
            raise ValueError("price_max debe ser mayor a 0")

        # ── PASO 1: Embedding de la consulta ──────────────────────────
        # Convertimos el texto del usuario en un vector de 384 dimensiones.
        # El mismo espacio matemático donde están los productos indexados.
        logger.info("Iniciando búsqueda RAG | query: '%s'", query)

        logger.info("Paso 1/3: Generando embedding de la consulta")
        query_embedding = self._embedding_service.embed_text(query)

        # ── PASO 2: Búsqueda por similitud en ChromaDB ────────────────
        # Construir filtros opcionales de metadata
        where_filter = self._build_where_filter(
            price_max=price_max,
            category=category,
        )

        logger.info("Paso 2/3: Buscando productos similares en ChromaDB")
        retrieved_products = self._vector_store.search(
            query_embedding=query_embedding,
            n_results=n_results,
            where=where_filter,
        )

        logger.info("Recuperados %d productos", len(retrieved_products))

        # ── PASO 3: Generación de respuesta con Gemini ────────────────
        # Gemini recibe la consulta original + los productos encontrados
        # y genera una respuesta en lenguaje natural con justificación.
        logger.info("Paso 3/3: Generando respuesta con Gemini")
        ai_response = self._gemini_service.generate_response(
            user_query=query,
            retrieved_products=retrieved_products,
        )

        result = {
            "query": query,
            "retrieved_products": retrieved_products,
            "ai_response": ai_response,
            "total_found": len(retrieved_products),
        }

        logger.info("Pipeline RAG completado exitosamente")
        return result

    def _build_where_filter(
        self,
        price_max: float | None,
        category: str | None,
    ) -> dict | None:
        """
        Construye el filtro de metadata para ChromaDB.

        ChromaDB usa un lenguaje de filtros similar a MongoDB.
        Si hay múltiples condiciones se combinan con $and.

        Ejemplos:
            price_max=200            → {"price": {"$lte": 200}}
            category="audífonos"     → {"category": {"$eq": "audífonos"}}
            ambos                    → {"$and": [{...}, {...}]}
            ninguno                  → None (sin filtro)
        """
        conditions = []

        if price_max is not None:
            conditions.append({"price": {"$lte": price_max}})

        if category is not None:
            conditions.append({"category": {"$eq": category}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]

        # Múltiples condiciones → operador $and
        return {"$and": conditions}