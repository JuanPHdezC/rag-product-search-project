from pydantic import BaseModel, Field


class FaithfulnessClaim(BaseModel):
    """
    Una afirmación individual extraída de la respuesta del LLM,
    junto con el veredicto de si está soportada por el contexto.
    """

    claim: str = Field(description="La afirmación extraída de la respuesta")
    supported: bool = Field(description="Si el contexto respalda esta afirmación")
    reasoning: str = Field(description="Por qué se considera soportada o no")


class FaithfulnessResult(BaseModel):
    """Resultado de evaluar Faithfulness sobre una respuesta."""

    claims: list[FaithfulnessClaim]
    score: float = Field(ge=0.0, le=1.0)

    @property
    def supported_count(self) -> int:
        return sum(1 for c in self.claims if c.supported)


class AnswerRelevanceResult(BaseModel):
    """Resultado de evaluar Answer Relevance sobre una respuesta."""

    score: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Justificación del score asignado")


class ContextPrecisionResult(BaseModel):
    """
    Resultado de evaluar Context Precision.
    Compara productos recuperados contra el ground truth anotado.
    """

    retrieved_ids: list[str]
    relevant_retrieved_ids: list[str] = Field(
        description="De los recuperados, cuáles SÍ son relevantes según ground truth"
    )
    score: float = Field(ge=0.0, le=1.0)


class ContextRecallResult(BaseModel):
    """
    Resultado de evaluar Context Recall.
    Compara contra todos los relevantes que existen en el catálogo.
    """

    total_relevant_in_catalog: int
    relevant_retrieved_count: int
    score: float = Field(ge=0.0, le=1.0)


class EvalCaseResult(BaseModel):
    """
    Resultado completo de evaluar un caso del golden dataset
    con las 4 métricas.
    """

    eval_id: str
    query: str
    faithfulness: FaithfulnessResult
    answer_relevance: AnswerRelevanceResult
    context_precision: ContextPrecisionResult
    context_recall: ContextRecallResult

    @property
    def average_score(self) -> float:
        """Score promedio de las 4 métricas — vista rápida de calidad general."""
        return round(
            (
                self.faithfulness.score
                + self.answer_relevance.score
                + self.context_precision.score
                + self.context_recall.score
            )
            / 4,
            4,
        )