import logging

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings

logger = logging.getLogger(__name__)

# Dimensiones del modelo all-MiniLM-L6-v2.
# Validar esto explícitamente protege contra mezclar embeddings
# de modelos distintos en la misma colección.
_EXPECTED_EMBEDDING_DIM = 384


class VectorStoreRepository:
    """
    Responsabilidad: servir como intermediario entre la aplicación y ChromaDB.

    Patrón Repository: el resto de la aplicación no sabe si usamos
    ChromaDB, FAISS o Pinecone. Solo conoce esta interfaz.Si en el futuro
    se migra a otro vector store, solo cambia este archivo.

    En producción este patrón es crítico. Una empresa puede tener
    diferentes vector stores para diferentes casos de uso (búsqueda
    de productos, recomendaciones, detección de fraude) todos detrás
    de la misma interfaz. Es un patrón de diseño fundamental para mantener
    la flexibilidad y la escalabilidad de la aplicación.
    """

    def __init__(self) -> None:
        logger.info("Inicializando ChromaDB en: %s", settings.chroma_persist_path)

        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "Colección '%s' lista con %d documentos",
            settings.chroma_collection_name,
            self._collection.count(),
        )

    def add_products(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """
        Indexa productos en ChromaDB.

        ChromaDB almacena 4 cosas por documento:
        - id: identificador único (para upsert y deduplicación)
        - embedding: el vector numérico (para búsqueda por similitud)
        - document: el texto original (para dárselo al LLM como contexto)
        - metadata: campos filtrables (precio, categoría, etc.)

        Usamos upsert en lugar de add:
        - add falla si el id ya existe
        - upsert actualiza si existe, crea si no, garantizando idempotencia.

        Raises:
            ValueError: Si las listas tienen longitudes inconsistentes
                        o los embeddings tienen dimensiones incorrectas.
        """
        if not ids:
            raise ValueError("La lista de ids no puede estar vacía")

        # Todas las listas deben tener la misma longitud — cada producto
        # necesita exactamente un id, un embedding, un documento y una metadata
        lengths = {"ids": len(ids), "embeddings": len(embeddings),
                   "documents": len(documents), "metadatas": len(metadatas)}

        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"Las listas deben tener la misma longitud. "
                f"Longitudes recibidas: {lengths}"
            )

        # Validar dimensiones del primer embedding como muestra representativa
        if embeddings and len(embeddings[0]) != _EXPECTED_EMBEDDING_DIM:
            raise ValueError(
                f"Embedding inválido: se esperaban {_EXPECTED_EMBEDDING_DIM} "
                f"dimensiones, se recibieron {len(embeddings[0])}. "
                f"¿Estás usando el modelo correcto?"
            )

        logger.info("Indexando %d productos en ChromaDB", len(ids))

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(
            "Indexación completada. Total en colección: %d",
            self._collection.count(),
        )

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Busca los productos más similares al embedding de consulta.

        Args:

            query_embedding: Vector de la consulta del usuario.

            n_results: Cuántos productos devolver (top-k).

            where: Filtro opcional por metadata.
                   Ejemplo: {"price": {"$lte": 200}} para precio <= 200
                   Ejemplo: {"category": "audífonos"}

        Returns:

            Lista de dicts con id, document, metadata y distance,
            ordenada de más a menos relevante.



        Por qué distance y no score:
        ChromaDB devuelve 'distance' con coseno: 0 = idéntico, 2 = opuesto.
        Lo convertimos a 'similarity_score' (1 - distance/2) para que
        sea más intuitivo: 1.0 = perfecto, 0.0 = sin relación.

        Raises:
            ValueError: Si el embedding tiene dimensiones incorrectas
                        o n_results es inválido.
        """
        if len(query_embedding) != _EXPECTED_EMBEDDING_DIM:
            raise ValueError(
                f"Embedding de consulta inválido: se esperaban "
                f"{_EXPECTED_EMBEDDING_DIM} dimensiones, "
                f"se recibieron {len(query_embedding)}"
            )

        if not 1 <= n_results <= 20:
            raise ValueError("n_results debe estar entre 1 y 20")

        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }

        if where:
            query_params["where"] = where

        results = self._collection.query(**query_params)

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        products = []
        for prod_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            products.append({
                "id": prod_id,
                "document": doc,
                "metadata": meta,
                "similarity_score": round(1 - dist / 2, 4),
            })

        return products

    def count(self) -> int:
        """Devuelve cuántos productos hay indexados."""
        return self._collection.count()

    def is_empty(self) -> bool:
        """Verifica si la colección está vacía."""
        return self._collection.count() == 0
    
    def get_by_id(self, product_id: str) -> dict | None:
        """
        Recupera un producto específico por su id exacto.

        A diferencia de search() que busca por similitud semántica,
        este método hace un lookup determinístico. Solo hay 2 caminos. 
        El producto existe con ese id exacto, o no existe. 
        No hay "aproximación".

        Se usa en QAService cuando el cliente ya sabe qué producto
        quiere consultar (ej: el usuario está viendo la ficha de un
        producto y hace una pregunta sobre él). Evita gastar una
        llamada de embedding+búsqueda innecesaria.

        Args:
            product_id: id exacto del producto (ej: "prod_001")

        Returns:
            Dict con id, document, metadata y similarity_score=1.0
            si el producto existe, None si no existe.
        """
        if not product_id or not product_id.strip():
            raise ValueError("product_id no puede estar vacío")

        results = self._collection.get(
            ids=[product_id],
            include=["documents", "metadatas"],
        )

        # ChromaDB devuelve listas vacías si el id no existe
        # Verificamos explícitamente(no lanza excepción) 
        if not results["ids"]:
            return None

        return {
            "id": results["ids"][0],
            "document": results["documents"][0],
            "metadata": results["metadatas"][0],
            "similarity_score": 1.0,  # lookup exacto = relevancia perfecta
        }