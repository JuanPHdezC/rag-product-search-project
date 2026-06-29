import json
import logging
import time

from google import genai
from google.genai import types

from app.core.config import settings
from app.core.gemini_retry import call_with_retry

from app.models.evaluation import (
    AnswerRelevanceResult,
    ContextPrecisionResult,
    ContextRecallResult,
    EvalCaseResult,
    FaithfulnessClaim,
    FaithfulnessResult,
)

logger = logging.getLogger(__name__)


class EvaluationService:
    """
    Implementa las métricas de evaluación RAG (Faithfulness,
    Answer Relevance, Context Precision, Context Recall) usando
    Gemini como LLM-as-judge.

    Por qué implementación propia en lugar de RAGAs:
    RAGAs 0.4.3 introduce 38 dependencias nuevas con conflictos
    de versiones (websockets, pydantic) y un bug interno
    (import roto de langchain_community.chat_models.vertexai).
    Para un catálogo de 12 productos, el costo de esas
    dependencias no es proporcional al valor que aportan.
    Ver DEVLOG.md Iteración 003 para el análisis completo.

    Cada método de evaluación usa un prompt distinto, diseñado
    para forzar una respuesta JSON parseable, permitiendo calcular 
    scores de forma determinística en lugar de depender de 
    interpretación de texto libre.
    """

    # temperature=0.0 en evaluación: queremos máxima consistencia
    # y reproducibilidad, no creatividad. Es lo opuesto al criterio
    # usado en GeminiService.generate_response (temperature=0.3),
    # donde sí queremos cierta naturalidad en el lenguaje.
    _JUDGE_TEMPERATURE = 0.0

    def __init__(self) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model
        logger.info("EvaluationService inicializado | modelo juez: %s", self._model)


    def evaluate_faithfulness(
        self,
        answer: str,
        context: str,
    ) -> FaithfulnessResult:
        """
        Mide si cada afirmación de la respuesta está soportada
        por el contexto recuperado (anti-alucinación).

        Proceso:
        1. Gemini descompone la respuesta en afirmaciones atómicas
        2. Para cada afirmación, verifica si el contexto la respalda
        3. Score = afirmaciones soportadas / total de afirmaciones
        """
        prompt = f"""Eres un evaluador experto en sistemas RAG. Tu tarea es 
        verificar si una respuesta generada está completamente respaldada 
        por el contexto proporcionado, sin inventar información.

        CONTEXTO (productos reales del catálogo):
        {context}

        RESPUESTA A EVALUAR:
        {answer}

        INSTRUCCIONES:
        1. Descompón la respuesta en afirmaciones atómicas individuales 
        (cada dato concreto: precio, característica, marca, etc.)
        2. Para cada afirmación, determina si está EXPLÍCITAMENTE 
        respaldada por el contexto
        3. Responde ÚNICAMENTE con un JSON válido, sin texto adicional, 
        con este formato exacto:

        {{
        "claims": [
            {{"claim": "texto de la afirmación", "supported": true, "reasoning": "por qué"}},
            {{"claim": "texto de la afirmación", "supported": false, "reasoning": "por qué"}}
        ]
        }}"""

        response_text = call_with_retry(
            fn=lambda: self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self._JUDGE_TEMPERATURE,
                    response_mime_type="application/json",
                ),
            ).text,
            context="evaluate_faithfulness",
        )
        data = json.loads(response_text)
        claims = [FaithfulnessClaim(**c) for c in data["claims"]]

        if not claims:
            # Sin afirmaciones verificables (caso borde, score neutro)
            return FaithfulnessResult(claims=[], score=1.0)

        supported = sum(1 for c in claims if c.supported)
        score = round(supported / len(claims), 4)

        return FaithfulnessResult(claims=claims, score=score)

    def evaluate_answer_relevance(
        self,
        query: str,
        answer: str,
    ) -> AnswerRelevanceResult:
        """
        Mide si la respuesta contesta directamente la consulta,
        independientemente de si la información es correcta
        (eso lo mide Faithfulness por separado).
        """
        prompt = f"""Eres un evaluador experto en sistemas RAG. Tu tarea es 
        calificar qué tan directamente una respuesta contesta la consulta 
        del usuario, en una escala de 0.0 a 1.0.

        CONSULTA DEL USUARIO:
        {query}

        RESPUESTA A EVALUAR:
        {answer}

        CRITERIOS:
        - 1.0: contesta directamente todos los aspectos de la consulta 
        (incluyendo restricciones como precio o características específicas)
        - 0.5: contesta parcialmente, omite restricciones importantes 
        de la consulta
        - 0.0: respuesta genérica que no aborda lo que se preguntó

        Responde ÚNICAMENTE con un JSON válido, sin texto adicional:
        {{"score": 0.0, "reasoning": "explicación breve"}}"""

        response_text = call_with_retry(
            fn=lambda: self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self._JUDGE_TEMPERATURE,
                    response_mime_type="application/json",
                ),
            ).text,
            context="evaluate_answer_relevance",
        )
        data = json.loads(response_text)
        return AnswerRelevanceResult(**data)

    def evaluate_context_precision(
        self,
        retrieved_ids: list[str],
        ground_truth_relevant_ids: list[str],
    ) -> ContextPrecisionResult:
        """
        De los productos recuperados por ChromaDB, ¿cuántos
        eran realmente relevantes según el ground truth anotado?

        A diferencia de Faithfulness y Answer Relevance, esta
        métrica NO necesita Gemini. Es un cálculo determinístico
        sobre IDs, comparando contra el golden dataset.
        """
        relevant_set = set(ground_truth_relevant_ids)
        relevant_retrieved = [
            pid for pid in retrieved_ids if pid in relevant_set
        ]

        if not retrieved_ids:
            score = 1.0 if not ground_truth_relevant_ids else 0.0
        else:
            score = round(len(relevant_retrieved) / len(retrieved_ids), 4)

        return ContextPrecisionResult(
            retrieved_ids=retrieved_ids,
            relevant_retrieved_ids=relevant_retrieved,
            score=score,
        )

    def evaluate_context_recall(
        self,
        retrieved_ids: list[str],
        ground_truth_relevant_ids: list[str],
    ) -> ContextRecallResult:
        """
        De todos los productos relevantes que existen en el
        catálogo para esta consulta, ¿cuántos se recuperaron?

        También es determinístico. Depende del golden dataset,
        no de Gemini.
        """
        if not ground_truth_relevant_ids:
            # Caso negativo (eval_010): no hay relevantes esperados.
            # Recall es perfecto trivialmente si no se recuperó
            # nada relevante (porque no existe nada relevante).
            return ContextRecallResult(
                total_relevant_in_catalog=0,
                relevant_retrieved_count=0,
                score=1.0,
            )

        retrieved_set = set(retrieved_ids)
        relevant_retrieved_count = sum(
            1 for rid in ground_truth_relevant_ids if rid in retrieved_set
        )

        score = round(
            relevant_retrieved_count / len(ground_truth_relevant_ids), 4
        )

        return ContextRecallResult(
            total_relevant_in_catalog=len(ground_truth_relevant_ids),
            relevant_retrieved_count=relevant_retrieved_count,
            score=score,
        )

    def evaluate_case(
        self,
        eval_id: str,
        query: str,
        answer: str,
        context: str,
        retrieved_ids: list[str],
        ground_truth_relevant_ids: list[str],
    ) -> EvalCaseResult:
        """
        Ejecuta las 4 métricas sobre un caso del golden dataset
        y devuelve el resultado consolidado.
        """
        logger.info("Evaluando caso: %s | query: '%s'", eval_id, query)

        faithfulness = self.evaluate_faithfulness(answer=answer, context=context)
        answer_relevance = self.evaluate_answer_relevance(query=query, answer=answer)
        context_precision = self.evaluate_context_precision(
            retrieved_ids=retrieved_ids,
            ground_truth_relevant_ids=ground_truth_relevant_ids,
        )
        context_recall = self.evaluate_context_recall(
            retrieved_ids=retrieved_ids,
            ground_truth_relevant_ids=ground_truth_relevant_ids,
        )

        return EvalCaseResult(
            eval_id=eval_id,
            query=query,
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_precision=context_precision,
            context_recall=context_recall,
        )