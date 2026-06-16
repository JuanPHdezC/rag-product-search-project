"""
Pipeline de evaluación offline del sistema RAG.

Ejecuta cada consulta del golden dataset contra el pipeline real,
evalúa las 4 métricas (Faithfulness, Answer Relevance,
Context Precision, Context Recall) y genera un reporte consolidado.

Se ejecuta manualmente o en CI/CD antes de deployar cambios y
permite responder: "¿esta versión del pipeline es mejor o peor
que la anterior?"
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.WARNING,  # se silencia INFO para que el reporte se lea limpio
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from app.core.telemetry import get_telemetry_client
from app.models.evaluation import EvalCaseResult
from app.repositories.vector_store import VectorStoreRepository
from app.services.embedding_service import get_embedding_service
from app.services.evaluation_service import EvaluationService
from app.services.gemini_service import get_gemini_service
from app.services.search_service import SearchService

GOLDEN_DATASET_PATH = (
    Path(__file__).parent.parent / "data" / "eval" / "golden_dataset.json"
)
GOLDEN_DATASET_SMOKE_PATH = (
    Path(__file__).parent.parent / "data" / "eval" / "golden_dataset_smoke.json"
)
REPORT_OUTPUT_PATH = (
    Path(__file__).parent.parent / "data" / "eval" / "last_report.md"
)


def load_golden_dataset(path: Path) -> list[dict]:
    """Carga el golden dataset anotado manualmente."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(
    golden_dataset: list[dict],
    search_service: SearchService,
    eval_service: EvaluationService,
) -> list[EvalCaseResult]:
    """
    Ejecuta el pipeline real para cada caso del golden dataset
    y evalúa las 4 métricas.
    """
    results: list[EvalCaseResult] = []

    for case in golden_dataset:
        eval_id = case["id"]
        query = case["query"]
        ground_truth_ids = case["relevant_product_ids"]

        print(f"Evaluando {eval_id}: '{query}'...")

        time.sleep(3)  # throttling proactivo para reducción de choques con el rate limit de Gemini

        # Ejecutar el pipeline real (mismo código que usa la API)
        search_result = search_service.search(query=query, n_results=5)

        retrieved_ids = [p["id"] for p in search_result["retrieved_products"]]
        context = "\n\n".join(
            p["document"] for p in search_result["retrieved_products"]
        )
        answer = search_result["ai_response"]

        # Caso especial: sin productos recuperados, Faithfulness/Relevance
        # no tienen contexto para evaluar contra; se evalúan igual.
        # El LLM debería haber dicho explícitamente que no hay resultados
        case_result = eval_service.evaluate_case(
            eval_id=eval_id,
            query=query,
            answer=answer,
            context=context if context else "(sin productos recuperados)",
            retrieved_ids=retrieved_ids,
            ground_truth_relevant_ids=ground_truth_ids,
        )

        results.append(case_result)

    return results


def build_markdown_report(results: list[EvalCaseResult]) -> str:
    """
    Genera un reporte markdown legible con resultados por caso
    y promedios agregados de cada métrica.
    """
    n = len(results)
    avg_faithfulness = round(sum(r.faithfulness.score for r in results) / n, 4)
    avg_relevance = round(sum(r.answer_relevance.score for r in results) / n, 4)
    avg_precision = round(sum(r.context_precision.score for r in results) / n, 4)
    avg_recall = round(sum(r.context_recall.score for r in results) / n, 4)

    lines = [
        "# Reporte de Evaluación RAG",
        "",
        f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Casos evaluados:** {n}",
        "",
        "## Métricas agregadas",
        "",
        "| Métrica | Promedio |",
        "|---|---|",
        f"| Faithfulness | {avg_faithfulness} |",
        f"| Answer Relevance | {avg_relevance} |",
        f"| Context Precision | {avg_precision} |",
        f"| Context Recall | {avg_recall} |",
        "",
        "## Resultados por caso",
        "",
    ]

    for r in results:
        lines.extend([
            f"### {r.eval_id} — \"{r.query}\"",
            "",
            f"| Métrica | Score |",
            f"|---|---|",
            f"| Faithfulness | {r.faithfulness.score} |",
            f"| Answer Relevance | {r.answer_relevance.score} |",
            f"| Context Precision | {r.context_precision.score} |",
            f"| Context Recall | {r.context_recall.score} |",
            f"| **Promedio** | **{r.average_score}** |",
            "",
        ])

        # Mostrar afirmaciones no soportadas para observabildiad detallada de las alucinaciones
        unsupported = [c for c in r.faithfulness.claims if not c.supported]
        if unsupported:
            lines.append("**⚠️ Afirmaciones no soportadas (posibles alucinaciones):**")
            for claim in unsupported:
                lines.append(f"- {claim.claim} — *{claim.reasoning}*")
            lines.append("")

    return "\n".join(lines)


def send_scores_to_langfuse(results: list[EvalCaseResult]) -> None:
    """
    Envía los scores de evaluación a LangFuse para visibilidad
    histórica en el dashboard, asociados como scores generales
    (no ligados a un trace específico, ya que esta evaluación
    corre fuera del contexto de un request real).
    """
    telemetry = get_telemetry_client()

    if not telemetry.is_enabled:
        logger.warning("LangFuse no habilitado — scores no enviados")
        return

    for r in results:
        telemetry.client.create_score(
            name="faithfulness",
            value=r.faithfulness.score,
            comment=f"eval_id={r.eval_id}",
        )
        telemetry.client.create_score(
            name="answer_relevance",
            value=r.answer_relevance.score,
            comment=f"eval_id={r.eval_id}",
        )
        telemetry.client.create_score(
            name="context_precision",
            value=r.context_precision.score,
            comment=f"eval_id={r.eval_id}",
        )
        telemetry.client.create_score(
            name="context_recall",
            value=r.context_recall.score,
            comment=f"eval_id={r.eval_id}",
        )

    telemetry.flush()
    print("\n✅ Scores enviados a LangFuse")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluación offline del pipeline RAG"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Usa el dataset reducido (5 casos) en lugar del completo (10 casos). "
             "Útil para validar el pipeline de evaluación sin agotar la cuota "
             "diaria gratuita de Gemini (20 requests/día).",
    )
    args = parser.parse_args()

    dataset_path = GOLDEN_DATASET_SMOKE_PATH if args.smoke else GOLDEN_DATASET_PATH
    mode_label = "SMOKE (5 casos)" if args.smoke else "COMPLETO (10 casos)"

    print(f"=== Iniciando evaluación offline del pipeline RAG — modo {mode_label} ===\n")

    golden_dataset = load_golden_dataset(dataset_path)
    print(f"Golden dataset cargado: {len(golden_dataset)} casos\n")
    print(f"Llamadas estimadas a Gemini: {len(golden_dataset) * 3} "
          f"(cuota gratuita diaria típica: 20)\n")

    search_service = SearchService(
        embedding_service=get_embedding_service(),
        vector_store=VectorStoreRepository(),
        gemini_service=get_gemini_service(),
        telemetry=get_telemetry_client(),
    )
    eval_service = EvaluationService()

    results = run_evaluation(golden_dataset, search_service, eval_service)

    report = build_markdown_report(results)
    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_PATH.write_text(report, encoding="utf-8")

    print(f"\n✅ Reporte generado en: {REPORT_OUTPUT_PATH}")
    print("\n" + "=" * 60)
    print(report)

    send_scores_to_langfuse(results)


if __name__ == "__main__":
    main()