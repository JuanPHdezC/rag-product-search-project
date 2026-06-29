"""
Script de ingesta: lee el catálogo JSON, genera embeddings
y los persiste en ChromaDB.

Se ejecuta una sola vez (o cuando el catálogo cambia).
No es parte de la API, es un proceso offline separado.

En producción esto sería un job programado (cron, Airflow, etc.)
que re-indexa cuando se actualizan los productos.
"""

import json
import logging
import sys
from pathlib import Path

# Agregar el directorio raíz al path para poder importar 'app'
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.product import Product
from app.repositories.vector_store import VectorStoreRepository
from app.services.embedding_service import get_embedding_service

# Configurar logging para observabilidad durante la indexación
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Path al catálogo de productos
PRODUCTS_PATH = Path(__file__).parent.parent / "data" / "raw" / "products.json"


def load_products(path: Path) -> list[Product]:
    """
    Carga y valida el catálogo de productos desde JSON.

    Pydantic valida cada producto al instanciarlo.
    Si un producto tiene datos malformados, falla aquí
    con un error prematuro, evitando errores en tiempo 
    de ejecución de la aplicación.
    """
    logger.info("Cargando catálogo desde: %s", path)

    with open(path, encoding="utf-8") as f:
        raw_data = json.load(f)

    products = [Product(**item) for item in raw_data]
    logger.info("Cargados %d productos válidos", len(products))
    return products


def index_products(products: list[Product]) -> None:
    """
    Orquesta el pipeline completo de indexación:
    1. Generar texto optimizado para embedding por producto
    2. Generar embeddings en batch
    3. Persistir en ChromaDB
    """
    embedding_service = get_embedding_service()
    vector_store = VectorStoreRepository()

    # Preparar los 4 componentes que ChromaDB necesita
    ids = [p.id for p in products]
    texts = [p.to_embedding_text() for p in products]
    metadatas = [p.to_chroma_metadata() for p in products]

    # Generar todos los embeddings en un solo batch
    logger.info("Generando embeddings para %d productos...", len(products))
    embeddings = embedding_service.embed_batch(texts)

    # Persistir en ChromaDB
    vector_store.add_products(
        ids=ids,
        embeddings=embeddings,
        documents=texts,    # guardamos el texto para dárselo a Gemini como contexto
        metadatas=metadatas,
    )

    logger.info("✅ Indexación completada. %d productos en vector store.", vector_store.count())


def main() -> None:
    logger.info("=== Iniciando pipeline de indexación ===")
    products = load_products(PRODUCTS_PATH)
    index_products(products)
    logger.info("=== Pipeline de indexación finalizado ===")


if __name__ == "__main__":
    main()