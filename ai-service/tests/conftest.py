"""
Fixtures compartidas entre todos los tests del proyecto.

pytest descubre conftest.py automáticamente permitiendo que
cualquier fixture definida aquí este disponible en todos los 
archivos test_*.py sin necesidad de import explícito.
"""

from unittest.mock import MagicMock

import pytest

from app.core.telemetry import TelemetryClient
from app.repositories.vector_store import VectorStoreRepository
from app.services.embedding_service import EmbeddingService
from app.services.gemini_service import GeminiService


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    """
    Doble de EmbeddingService que devuelve un vector fijo de 384
    dimensiones sin cargar el modelo real de sentence-transformers.

    spec=EmbeddingService hace que el mock solo acepte llamadas
    a métodos que existen realmente en la clase. Si el código
    de producción cambia el nombre de un método, el test falla
    aquí en lugar de fallar silenciosamente.
    """
    mock = MagicMock(spec=EmbeddingService)
    # Vector de 384 floats, mismo formato que produce el modelo real
    mock.embed_text.return_value = [0.1] * 384
    return mock


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """
    Doble de VectorStoreRepository que devuelve productos fijos
    sin necesitar una conexión real a ChromaDB en disco.
    """
    mock = MagicMock(spec=VectorStoreRepository)
    mock.search.return_value = [
        {
            "id": "prod_001",
            "document": "Sony WH-1000XM5. Categoría: audífonos...",
            "metadata": {
                "product_id": "prod_001",
                "name": "Sony WH-1000XM5",
                "category": "audífonos",
                "brand": "Sony",
                "price": 179.99,
                "currency": "USD",
                "tags": "inalámbrico, cancelación de ruido",
            },
            "similarity_score": 0.95,
        },
    ]
    mock.count.return_value = 12
    mock.is_empty.return_value = False
    return mock


@pytest.fixture
def mock_gemini_service() -> MagicMock:
    """
    Doble de GeminiService que devuelve una respuesta fija de
    texto sin llamar a la API real de Gemini (sin costo, sin
    latencia de red, sin riesgo de rate limit en tests).
    """
    mock = MagicMock(spec=GeminiService)
    mock.generate_response.return_value = (
        "El Sony WH-1000XM5 es una excelente opción con "
        "cancelación de ruido líder en la industria."
    )
    # _model es un atributo, no un método. Se accede directo
    mock._model = "gemini-2.5-flash"
    return mock


@pytest.fixture
def mock_telemetry_disabled() -> MagicMock:
    """
    Doble de TelemetryClient con observabilidad deshabilitada.

    Por qué deshabilitada y no mockeada con datos falsos:
    los tests no deben generar traces reales en LangFuse —
    contaminaría el dashboard con datos de prueba mezclados
    con datos reales de desarrollo/producción.
    """
    mock = MagicMock(spec=TelemetryClient)
    mock.is_enabled = False
    mock.client = None
    # observation() debe comportarse como no-op (context manager
    # que no hace nada), igual que el comportamiento real cuando
    # LangFuse no está habilitado
    mock.observation.return_value.__enter__ = MagicMock(return_value=None)
    mock.observation.return_value.__exit__ = MagicMock(return_value=False)
    return mock