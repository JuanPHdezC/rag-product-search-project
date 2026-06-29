"""
Script de prueba del pipeline RAG completo.
Permite verificar que todo funciona antes de exponer la API.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from app.repositories.vector_store import VectorStoreRepository
from app.services.embedding_service import get_embedding_service
from app.services.gemini_service import get_gemini_service
from app.services.search_service import SearchService


def run_test(query: str, price_max: float | None = None) -> None:
    print("\n" + "═" * 60)
    print(f"CONSULTA: {query}")
    if price_max:
        print(f"FILTRO:   precio máximo ${price_max}")
    print("═" * 60)

    # Instanciar el pipeline con dependency injection
    service = SearchService(
        embedding_service=get_embedding_service(),
        vector_store=VectorStoreRepository(),
        gemini_service=get_gemini_service(),
    )

    result = service.search(query=query, n_results=3, price_max=price_max)

    print(f"\n📦 PRODUCTOS RECUPERADOS ({result['total_found']}):")
    for p in result["retrieved_products"]:
        meta = p["metadata"]
        print(f"  • {meta['name']} — ${meta['price']} (score: {p['similarity_score']})")

    print(f"\n🤖 RESPUESTA DE GEMINI:\n{result['ai_response']}")


if __name__ == "__main__":
    # Prueba 1: búsqueda semántica básica
    run_test("audífonos inalámbricos con cancelación de ruido")

    # Prueba 2: búsqueda con filtro de precio
    run_test("audífonos inalámbricos con cancelación de ruido", price_max=150)

    # Prueba 3: consulta en inglés (debe funcionar igual)
    run_test("wireless headphones noise cancelling")