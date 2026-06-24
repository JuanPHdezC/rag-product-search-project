from functools import lru_cache

from app.core.telemetry import get_telemetry_client
from app.repositories.vector_store import VectorStoreRepository
from app.services.embedding_service import get_embedding_service
from app.services.gemini_service import get_gemini_service
from app.services.search_service import SearchService
from app.services.qa_service import QAService


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStoreRepository:
    """
    Instancia única de VectorStoreRepository.

    VectorStoreRepository abre una conexión a ChromaDB al instanciarse.
    lru_cache permite mantener abierta la conexión durante toda la vida 
    de la aplicación, evitando la apertura/cierre de la conexión en cada request.
    """
    return VectorStoreRepository()


def get_search_service() -> SearchService:
    """
    Factory del SearchService con todas sus dependencias inyectadas.

    FastAPI llama esta función automáticamente cuando un endpoint
    declara 'service: SearchService = Depends(get_search_service)'.

    SearchService no requiere lru_cache debido a que es lightweight;
    solo orquesta referencias a servicios que ya están en caché.
    Crear una nueva instancia por request es barato y más seguro
    para evitar estado compartido entre requests concurrentes.

    TelemetryClient no tiene lru_cache aquí porque ya lo tiene
    internamente en get_telemetry_client(). Llamarlo múltiples
    veces siempre devuelve la misma instancia.
    """
    vector_store = get_vector_store()

    if vector_store.is_empty():
        raise RuntimeError(
            "El vector store está vacío."
            "Ejecuta primero la indexación de los productos con scripts/index_products.py"
        )

    return SearchService(
        embedding_service=get_embedding_service(),
        vector_store=vector_store,
        gemini_service=get_gemini_service(),
        telemetry=get_telemetry_client(),
    )

@lru_cache(maxsize=1)
def get_qa_service_instance() -> QAService:
    """
    Instancia única de QAService.
    """
    return QAService(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
        gemini_service=get_gemini_service(),
        telemetry=get_telemetry_client(),
    )

def get_qa_service() -> QAService:
    """Factory para FastAPI Depends."""
    return get_qa_service_instance()