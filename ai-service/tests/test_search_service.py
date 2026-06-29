"""
Tests del orquestador SearchService.

Usa las fixtures de conftest.py (mock_embedding_service,
mock_vector_store, mock_gemini_service, mock_telemetry_disabled)
para probar el flujo completo de search() sin llamar los
servicios implementados.
"""

import pytest

from app.services.search_service import SearchService


@pytest.fixture
def search_service(
    mock_embedding_service,
    mock_vector_store,
    mock_gemini_service,
    mock_telemetry_disabled,
) -> SearchService:
    """
    SearchService ensamblado con las 4 dependencias mockeadas.

    Esta es la instanciación directa mencionada en la planificación:
    nunca pasamos por get_embedding_service() ni las demás factory
    functions con lru_cache. Esto evita por completo el problema
    de contaminación de caché entre tests.
    """
    return SearchService(
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        gemini_service=mock_gemini_service,
        telemetry=mock_telemetry_disabled,
    )


class TestSearchServiceValidation:
    """Tests de las validaciones defensivas al inicio de search()."""

    def test_empty_query_raises_value_error(self, search_service):
        with pytest.raises(ValueError, match="no puede estar vacía"):
            search_service.search(query="")

    def test_whitespace_only_query_raises_value_error(self, search_service):
        with pytest.raises(ValueError, match="no puede estar vacía"):
            search_service.search(query="   ")

    def test_n_results_below_minimum_raises_value_error(self, search_service):
        with pytest.raises(ValueError, match="entre 1 y 20"):
            search_service.search(query="audífonos", n_results=0)

    def test_n_results_above_maximum_raises_value_error(self, search_service):
        with pytest.raises(ValueError, match="entre 1 y 20"):
            search_service.search(query="audífonos", n_results=21)

    def test_negative_price_max_raises_value_error(self, search_service):
        with pytest.raises(ValueError, match="mayor a 0"):
            search_service.search(query="audífonos", price_max=-10)


class TestSearchServiceOrchestration:
    """
    Tests del flujo completo. Verifican que SearchService llama
    correctamente a cada dependencia en el orden esperado y
    propaga los datos entre pasos sin perderlos.
    """

    def test_search_calls_embedding_service_with_the_query(
        self, search_service, mock_embedding_service
    ):
        """
        El primer paso del pipeline debe pasar exactamente el
        texto de la consulta al servicio de embeddings, sin
        modificarlo.
        """
        search_service.search(query="audífonos inalámbricos")

        mock_embedding_service.embed_text.assert_called_once_with(
            "audífonos inalámbricos"
        )

    def test_search_passes_embedding_to_vector_store(
        self, search_service, mock_vector_store, mock_embedding_service
    ):
        """
        El vector generado en el paso 1 debe ser exactamente el
        que se pasa a vector_store.search() en el paso 2. Verifica
        que los datos fluyen correctamente entre pasos.
        """
        search_service.search(query="audífonos inalámbricos")

        called_kwargs = mock_vector_store.search.call_args.kwargs
        assert called_kwargs["query_embedding"] == [0.1] * 384

    def test_search_returns_expected_result_structure(self, search_service):
        """
        El resultado final debe tener exactamente las 4 keys
        documentadas en el docstring de search(): query,
        retrieved_products, ai_response, total_found.
        """
        result = search_service.search(query="audífonos inalámbricos")

        assert set(result.keys()) == {
            "query",
            "retrieved_products",
            "ai_response",
            "total_found",
            "trace_id",
        }
        assert result["query"] == "audífonos inalámbricos"
        assert result["total_found"] == 1  # según el mock_vector_store
        assert result["trace_id"] is None  # telemetry deshabilitada

    def test_search_passes_retrieved_products_to_gemini(
        self, search_service, mock_gemini_service
    ):
        """
        Gemini debe recibir los productos recuperados por
        ChromaDB como contexto. Verifica la integración RAG
        completa: retrieval alimentando generation.
        """
        search_service.search(query="audífonos inalámbricos")

        called_kwargs = mock_gemini_service.generate_response.call_args.kwargs
        assert len(called_kwargs["retrieved_products"]) == 1
        assert called_kwargs["retrieved_products"][0]["id"] == "prod_001"


class TestSearchServiceFilters:
    """Tests de _build_where_filter. Lógica pura, sin mocks necesarios."""

    def test_no_filters_returns_none(self, search_service):
        result = search_service._build_where_filter(
            price_max=None, category=None
        )
        assert result is None

    def test_only_price_filter(self, search_service):
        result = search_service._build_where_filter(
            price_max=200.0, category=None
        )
        assert result == {"price": {"$lte": 200.0}}

    def test_only_category_filter(self, search_service):
        result = search_service._build_where_filter(
            price_max=None, category="audífonos"
        )
        assert result == {"category": {"$eq": "audífonos"}}

    def test_both_filters_combined_with_and(self, search_service):
        result = search_service._build_where_filter(
            price_max=200.0, category="audífonos"
        )
        assert result == {
            "$and": [
                {"price": {"$lte": 200.0}},
                {"category": {"$eq": "audífonos"}},
            ]
        }