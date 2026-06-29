"""
Tests de QAService
El orquestador del pipeline Q&A.

Manejo de dependencias mockeadas vía DI, 
sin tocar Gemini, ChromaDB ni embeddings reales.
"""

import pytest
from unittest.mock import MagicMock, patch

from app.services.qa_service import QAService


@pytest.fixture
def qa_service(
    mock_embedding_service,
    mock_vector_store,
    mock_gemini_service,
    mock_telemetry_disabled,
) -> QAService:
    """QAService con las 4 dependencias mockeadas."""
    return QAService(
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        gemini_service=mock_gemini_service,
        telemetry=mock_telemetry_disabled,
    )


class TestQAServiceValidation:
    def test_empty_question_raises_value_error(self, qa_service):
        with pytest.raises(ValueError, match="no puede estar vacía"):
            qa_service.answer(question="")

    def test_whitespace_only_question_raises_value_error(self, qa_service):
        with pytest.raises(ValueError, match="no puede estar vacía"):
            qa_service.answer(question="   ")

    def test_nonexistent_product_id_raises_value_error(
        self, qa_service, mock_vector_store
    ):
        """
        Si el cliente pasa un product_id que no existe en ChromaDB,
        el servicio debe fallar en lugar de intentar generar una 
        respuesta sin contexto.
        """
        mock_vector_store.get_by_id.return_value = None

        with pytest.raises(ValueError, match="No se encontró ningún producto"):
            qa_service.answer(
                question="¿cuántas horas de batería tiene?",
                product_id="prod_999",
            )


class TestQAServiceDirectLookup:
    """Tests del flujo con product_id explícito."""

    def test_uses_get_by_id_when_product_id_provided(
        self, qa_service, mock_vector_store
    ):
        """
        Con product_id en el request, el servicio debe usar
        get_by_id() y NO llamar a embed_text() ni search(),
        eso sería una llamada innecesaria que consume recursos.
        """
        with patch.object(qa_service, "_generate_answer", return_value="respuesta"):
            qa_service.answer(
                question="¿cuántas horas de batería tiene?",
                product_id="prod_001",
            )

        mock_vector_store.get_by_id.assert_called_once_with("prod_001")
        qa_service._embedding_service.embed_text.assert_not_called()
        mock_vector_store.search.assert_not_called()

    def test_result_contains_direct_id_as_found_via(
        self, qa_service, mock_vector_store
    ):
        with patch.object(qa_service, "_generate_answer", return_value="respuesta"):
            result = qa_service.answer(
                question="¿cuántas horas de batería tiene?",
                product_id="prod_001",
            )

        assert result["product_found_via"] == "direct_id"


class TestQAServiceSemanticSearch:
    """Tests del flujo de inferencia semántica (sin product_id)."""

    def test_uses_embedding_and_search_when_no_product_id(
        self, qa_service, mock_embedding_service, mock_vector_store
    ):
        """
        Sin product_id, el servicio debe generar un embedding
        de la pregunta y buscar el producto más relevante.
        """
        with patch.object(qa_service, "_generate_answer", return_value="respuesta"):
            qa_service.answer(
                question="¿el Sony WH-1000XM5 tiene cancelación de ruido?"
            )

        mock_embedding_service.embed_text.assert_called_once()
        mock_vector_store.search.assert_called_once()
        mock_vector_store.get_by_id.assert_not_called()

    def test_result_contains_semantic_search_as_found_via(
        self, qa_service
    ):
        with patch.object(qa_service, "_generate_answer", return_value="respuesta"):
            result = qa_service.answer(
                question="¿el Sony WH-1000XM5 tiene cancelación de ruido?"
            )

        assert result["product_found_via"] == "semantic_search"


class TestQAServiceResult:
    """Tests de la estructura del resultado."""

    def test_result_contains_all_expected_keys(self, qa_service):
        with patch.object(qa_service, "_generate_answer", return_value="respuesta"):
            result = qa_service.answer(
                question="¿cuántas horas de batería tiene?",
                product_id="prod_001",
            )

        assert set(result.keys()) == {
            "question",
            "answer",
            "product_id",
            "product_name",
            "source_document",
            "product_found_via",
            "trace_id",
        }

    def test_source_document_comes_from_vector_store_not_llm(
        self, qa_service, mock_vector_store
    ):
        """
        source_document debe ser el documento real de ChromaDB,
        nunca algo generado por el LLM, garantizando garantía 
        anti-alucinación.
        """
        with patch.object(qa_service, "_generate_answer", return_value="respuesta"):
            result = qa_service.answer(
                question="¿cuántas horas de batería tiene?",
                product_id="prod_001",
            )

        expected_document = mock_vector_store.get_by_id.return_value["document"]
        assert result["source_document"] == expected_document