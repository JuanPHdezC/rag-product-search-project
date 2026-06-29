"""
Tests de EvaluationService
Solo las métricas determinísticas (Context Precision, Context Recall), 
que no requieren LLM.

evaluate_faithfulness y evaluate_answer_relevance quedan fuera
de alcance de tests unitarios. Dependen de Gemini real para tener 
sentido semántico, se consideran tests de integración.
"""

import pytest

from app.services.evaluation_service import EvaluationService


@pytest.fixture
def eval_service() -> EvaluationService:
    """
    EvaluationService real (no mock)
    Solo se prueban los métodos que no llaman a self._client 
    (Gemini). No se requiere mockear el cliente para estos 
    tests específicos.
    """
    return EvaluationService.__new__(EvaluationService)
    # __new__ evita ejecutar __init__, que inicializaría un
    # cliente real de Gemini innecesario para estos tests ya que
    # los métodos que probamos no usan self._client.


class TestContextPrecision:
    """
    Context Precision: de los productos recuperados, ¿cuántos
    eran realmente relevantes según el ground truth?
    """

    def test_all_retrieved_are_relevant_gives_perfect_score(self, eval_service):
        result = eval_service.evaluate_context_precision(
            retrieved_ids=["prod_001", "prod_002"],
            ground_truth_relevant_ids=["prod_001", "prod_002", "prod_003"],
        )
        assert result.score == 1.0

    def test_none_retrieved_are_relevant_gives_zero_score(self, eval_service):
        result = eval_service.evaluate_context_precision(
            retrieved_ids=["prod_999"],
            ground_truth_relevant_ids=["prod_001", "prod_002"],
        )
        assert result.score == 0.0

    def test_partial_match_gives_proportional_score(self, eval_service):
        """
        2 de 4 recuperados son relevantes → precision = 0.5.
        Caso representativo del eval_008 (consulta ambigua)
        del golden dataset real.
        """
        result = eval_service.evaluate_context_precision(
            retrieved_ids=["prod_001", "prod_002", "prod_003", "prod_004"],
            ground_truth_relevant_ids=["prod_001", "prod_002"],
        )
        assert result.score == 0.5
        assert result.relevant_retrieved_ids == ["prod_001", "prod_002"]

    def test_empty_retrieval_with_no_expected_relevant_gives_perfect_score(
        self, eval_service
    ):
        """
        Caso negativo (como eval_010 del golden dataset): si no
        hay productos relevantes esperados y no se recuperó nada,
        el sistema acertó correctamente (score perfecto).
        """
        result = eval_service.evaluate_context_precision(
            retrieved_ids=[],
            ground_truth_relevant_ids=[],
        )
        assert result.score == 1.0

    def test_empty_retrieval_with_expected_relevant_gives_zero_score(
        self, eval_service
    ):
        """
        Si sí había productos relevantes esperados pero no se
        recuperó nada, el sistema falló completamente.
        """
        result = eval_service.evaluate_context_precision(
            retrieved_ids=[],
            ground_truth_relevant_ids=["prod_001"],
        )
        assert result.score == 0.0


class TestContextRecall:
    """
    Context Recall: de todos los productos relevantes que existen,
    ¿cuántos se recuperaron?
    """

    def test_all_relevant_retrieved_gives_perfect_score(self, eval_service):
        result = eval_service.evaluate_context_recall(
            retrieved_ids=["prod_001", "prod_002", "prod_999"],
            ground_truth_relevant_ids=["prod_001", "prod_002"],
        )
        assert result.score == 1.0

    def test_none_relevant_retrieved_gives_zero_score(self, eval_service):
        result = eval_service.evaluate_context_recall(
            retrieved_ids=["prod_999"],
            ground_truth_relevant_ids=["prod_001", "prod_002"],
        )
        assert result.score == 0.0

    def test_partial_recall_gives_proportional_score(self, eval_service):
        """
        1 de 2 relevantes esperados fue recuperado → recall = 0.5.
        """
        result = eval_service.evaluate_context_recall(
            retrieved_ids=["prod_001"],
            ground_truth_relevant_ids=["prod_001", "prod_002"],
        )
        assert result.score == 0.5
        assert result.relevant_retrieved_count == 1
        assert result.total_relevant_in_catalog == 2

    def test_negative_case_with_no_expected_relevant_gives_perfect_score(
        self, eval_service
    ):
        """
        Caso negativo (eval_010): no hay nada relevante que
        encontrar, entonces recall es trivialmente perfecto.
        """
        result = eval_service.evaluate_context_recall(
            retrieved_ids=["prod_001"],  # el sistema recuperó algo
            ground_truth_relevant_ids=[],  # pero nada era relevante
        )
        assert result.score == 1.0
        assert result.total_relevant_in_catalog == 0