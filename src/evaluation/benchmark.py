"""
Benchmark runner for evaluating RAG pipeline quality.

Defines benchmark suites with ground truth answers, runs queries through
the RAG pipeline, evaluates responses, and produces aggregate reports.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

from src.evaluation.metrics import RAGEvaluator, EvaluationResult, compute_rouge_l

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkCase:
    """A single benchmark test case"""
    query: str
    reference_answer: str
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseResult:
    """Evaluation result for a single benchmark case"""
    case: BenchmarkCase
    generated_answer: str
    context: str
    chunks: List[str]
    evaluation: EvaluationResult
    strategy: str = ""
    query_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.case.query,
            "reference_answer": self.case.reference_answer,
            "generated_answer": self.generated_answer,
            "category": self.case.category,
            "strategy": self.strategy,
            "query_time": self.query_time,
            "evaluation": self.evaluation.to_dict(),
        }


@dataclass
class BenchmarkReport:
    """Aggregate benchmark results"""
    name: str
    results: List[CaseResult]
    total_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_cases(self) -> int:
        return len(self.results)

    def _mean(self, values: List[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def aggregate_scores(self, category: Optional[str] = None) -> Dict[str, float]:
        """Compute mean scores, optionally filtered by category."""
        results = self.results
        if category is not None:
            results = [r for r in results if r.case.category == category]

        if not results:
            return {
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_precision": 0.0,
                "aggregate": 0.0,
                "rouge_l_f1": 0.0,
                "count": 0,
            }

        faith = [r.evaluation.faithfulness.score for r in results]
        relevance = [r.evaluation.answer_relevance.score for r in results]
        precision = [r.evaluation.context_precision.score for r in results]
        aggregate = [r.evaluation.aggregate_score for r in results]
        rouge_f1 = [
            r.evaluation.rouge_l.f1
            for r in results
            if r.evaluation.rouge_l is not None
        ]

        return {
            "faithfulness": self._mean(faith),
            "answer_relevance": self._mean(relevance),
            "context_precision": self._mean(precision),
            "aggregate": self._mean(aggregate),
            "rouge_l_f1": self._mean(rouge_f1),
            "count": len(results),
        }

    @property
    def categories(self) -> List[str]:
        return sorted({r.case.category for r in self.results})

    def to_dict(self) -> Dict[str, Any]:
        report = {
            "name": self.name,
            "total_time": self.total_time,
            "num_cases": self.num_cases,
            "overall_scores": self.aggregate_scores(),
            "scores_by_category": {
                cat: self.aggregate_scores(cat) for cat in self.categories
            },
            "results": [r.to_dict() for r in self.results],
            "metadata": self.metadata,
        }
        return report

    def save(self, path: str) -> None:
        """Save report to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Benchmark report saved to {path}")

    def summary(self) -> str:
        """Human-readable summary of the benchmark results."""
        lines = [
            f"Benchmark: {self.name}",
            f"Cases: {self.num_cases} | Time: {self.total_time:.1f}s",
            "",
            "Overall Scores:",
        ]
        overall = self.aggregate_scores()
        lines.append(f"  Faithfulness:       {overall['faithfulness']:.3f}")
        lines.append(f"  Answer Relevance:   {overall['answer_relevance']:.3f}")
        lines.append(f"  Context Precision:  {overall['context_precision']:.3f}")
        lines.append(f"  Aggregate:          {overall['aggregate']:.3f}")
        if overall["rouge_l_f1"] > 0:
            lines.append(f"  ROUGE-L F1:         {overall['rouge_l_f1']:.3f}")

        if len(self.categories) > 1:
            lines.append("")
            lines.append("Scores by Category:")
            for cat in self.categories:
                scores = self.aggregate_scores(cat)
                lines.append(
                    f"  {cat}: aggregate={scores['aggregate']:.3f} "
                    f"(n={scores['count']})"
                )

        return "\n".join(lines)


class BenchmarkRunner:
    """
    Runs benchmark suites against a RAG pipeline and evaluates results.

    Usage:
        rag = OllamaRAG(...)
        rag.load_pdf("paper.pdf")

        runner = BenchmarkRunner(evaluator_model="qwen2.5:14b")
        report = runner.run(rag, BERT_BENCHMARK, name="BERT evaluation")
        report.save("results/bert_benchmark.json")
        print(report.summary())
    """

    def __init__(
        self,
        evaluator_model: Optional[str] = None,
        evaluator_temperature: float = 0.1,
        timeout: Optional[int] = None,
    ):
        self.evaluator = RAGEvaluator(
            model=evaluator_model,
            temperature=evaluator_temperature,
            timeout=timeout,
        )

    def run(
        self,
        rag_instance: Any,
        cases: List[BenchmarkCase],
        name: str = "benchmark",
        strategy: str = "",
        categories: Optional[List[str]] = None,
    ) -> BenchmarkReport:
        """
        Run a benchmark suite against a RAG instance.

        Args:
            rag_instance: A RAG object with a query() method that returns
                          a dict with at least 'answer' and 'context' keys.
            cases: List of BenchmarkCase to evaluate.
            name: Name for this benchmark run.
            strategy: RAG strategy label (for comparison reports).
            categories: If provided, only run cases matching these categories.

        Returns:
            BenchmarkReport with per-case and aggregate results.
        """
        if categories:
            cases = [c for c in cases if c.category in categories]

        if not cases:
            logger.warning("No benchmark cases to run")
            return BenchmarkReport(name=name, results=[])

        logger.info(f"Running benchmark '{name}' with {len(cases)} cases")
        start_time = time.time()
        results = []

        for i, case in enumerate(cases):
            logger.info(
                f"  [{i + 1}/{len(cases)}] {case.category}: "
                f"{case.query[:60]}..."
            )

            try:
                query_start = time.time()
                rag_response = rag_instance.query(case.query)
                query_time = time.time() - query_start
            except Exception as e:
                logger.error(f"  Query failed: {e}")
                results.append(
                    CaseResult(
                        case=case,
                        generated_answer=f"ERROR: {e}",
                        context="",
                        chunks=[],
                        evaluation=_error_evaluation(str(e)),
                        strategy=strategy,
                        query_time=0.0,
                    )
                )
                continue

            # Handle both string and dict return types
            if isinstance(rag_response, dict):
                answer = rag_response.get("answer", "")
                context = rag_response.get("context", "")
                chunks = rag_response.get("chunks", None)
            else:
                answer = str(rag_response)
                context = ""
                chunks = None

            # Build context from retrieved docs if available
            if not context and hasattr(rag_instance, '_retrieve_documents'):
                try:
                    docs = rag_instance._retrieve_documents(case.query)
                    if docs:
                        context = "\n\n".join(
                            doc.page_content[:1000] for doc in docs
                        )
                except Exception:
                    pass

            if chunks is None:
                chunks = [c.strip() for c in context.split("\n\n") if c.strip()]

            evaluation = self.evaluator.evaluate_response(
                question=case.query,
                answer=answer,
                context=context,
                chunks=chunks,
                reference_answer=case.reference_answer,
            )

            results.append(
                CaseResult(
                    case=case,
                    generated_answer=answer,
                    context=context,
                    chunks=chunks,
                    evaluation=evaluation,
                    strategy=strategy,
                    query_time=query_time,
                )
            )

            logger.info(
                f"    Scores: faith={evaluation.faithfulness.score:.2f} "
                f"rel={evaluation.answer_relevance.score:.2f} "
                f"prec={evaluation.context_precision.score:.2f} "
                f"agg={evaluation.aggregate_score:.2f}"
            )

        total_time = time.time() - start_time

        report = BenchmarkReport(
            name=name,
            results=results,
            total_time=total_time,
            metadata={"strategy": strategy, "model": self.evaluator.model},
        )

        logger.info(f"Benchmark '{name}' completed in {total_time:.1f}s")
        return report

    @staticmethod
    def compare(reports: List[BenchmarkReport]) -> Dict[str, Any]:
        """
        Compare multiple benchmark reports side by side.

        Args:
            reports: List of BenchmarkReport objects to compare.

        Returns:
            Dictionary with per-strategy aggregate scores.
        """
        comparison = {}
        for report in reports:
            label = report.metadata.get("strategy", report.name)
            comparison[label] = {
                "overall": report.aggregate_scores(),
                "by_category": {
                    cat: report.aggregate_scores(cat)
                    for cat in report.categories
                },
                "total_time": report.total_time,
                "num_cases": report.num_cases,
            }
        return comparison


def _error_evaluation(error_msg: str) -> EvaluationResult:
    """Create an EvaluationResult representing a failed query."""
    from src.evaluation.metrics import MetricResult

    err = MetricResult(score=0.0, reasoning="", error=error_msg)
    return EvaluationResult(
        faithfulness=err,
        answer_relevance=err,
        context_precision=err,
    )


# ---------------------------------------------------------------------------
# Pre-built benchmark suites
# ---------------------------------------------------------------------------

ATTENTION_BENCHMARK: List[BenchmarkCase] = [
    BenchmarkCase(
        query="What is the Transformer architecture and what are its main components?",
        reference_answer=(
            "The Transformer is a sequence transduction model based entirely on "
            "attention mechanisms, dispensing with recurrence and convolutions. "
            "Its main components are an encoder and decoder, each composed of "
            "stacked layers with multi-head self-attention and position-wise "
            "feed-forward sub-layers, plus residual connections and layer "
            "normalization."
        ),
        category="architecture",
    ),
    BenchmarkCase(
        query="How does multi-head attention work in the Transformer?",
        reference_answer=(
            "Multi-head attention linearly projects queries, keys, and values "
            "h times with different learned projections, runs scaled dot-product "
            "attention on each projection in parallel, concatenates the results, "
            "and applies a final linear projection. This allows the model to "
            "attend to information from different representation subspaces at "
            "different positions."
        ),
        category="mechanism",
    ),
    BenchmarkCase(
        query="What is scaled dot-product attention and why is it scaled?",
        reference_answer=(
            "Scaled dot-product attention computes attention weights as "
            "softmax(QK^T / sqrt(d_k)) V, where Q, K, V are queries, keys, "
            "and values. The scaling factor 1/sqrt(d_k) prevents the dot "
            "products from growing large in magnitude for high dimensions, "
            "which would push the softmax into regions with extremely small "
            "gradients."
        ),
        category="mechanism",
    ),
    BenchmarkCase(
        query="What positional encoding is used and why is it needed?",
        reference_answer=(
            "The Transformer uses sinusoidal positional encodings added to "
            "the input embeddings. Sine and cosine functions of different "
            "frequencies encode each position. This is needed because the "
            "model contains no recurrence or convolution, so positional "
            "encodings inject information about token order."
        ),
        category="mechanism",
    ),
    BenchmarkCase(
        query=(
            "How does the Transformer's performance on English-to-German "
            "translation compare to previous state-of-the-art models?"
        ),
        reference_answer=(
            "The big Transformer model outperforms all previously published "
            "models and ensembles on the WMT 2014 English-to-German task, "
            "achieving a BLEU score of 28.4, which is more than 2 BLEU above "
            "the previous best."
        ),
        category="results",
    ),
    BenchmarkCase(
        query=(
            "What regularization techniques does the Transformer use during training?"
        ),
        reference_answer=(
            "The Transformer applies residual dropout to the output of each "
            "sub-layer before it is added to the sub-layer input and "
            "normalized, as well as to the sums of the embeddings and "
            "positional encodings. Label smoothing of value 0.1 is also used "
            "during training."
        ),
        category="training",
    ),
]


BERT_BENCHMARK: List[BenchmarkCase] = [
    BenchmarkCase(
        query="What is BERT and what does the name stand for?",
        reference_answer=(
            "BERT stands for Bidirectional Encoder Representations from "
            "Transformers. It is a language representation model designed to "
            "pre-train deep bidirectional representations from unlabeled text "
            "by jointly conditioning on both left and right context in all layers."
        ),
        category="overview",
    ),
    BenchmarkCase(
        query="What are the two pre-training objectives used by BERT?",
        reference_answer=(
            "BERT uses two pre-training objectives: masked language modeling "
            "(MLM), where random tokens are masked and the model predicts them "
            "from context, and next sentence prediction (NSP), where the model "
            "predicts whether two segments appear consecutively in the corpus."
        ),
        category="pretraining",
    ),
    BenchmarkCase(
        query="How does BERT's masked language model differ from standard language model pre-training?",
        reference_answer=(
            "Standard language models are trained left-to-right or "
            "right-to-left, conditioning only on one direction. BERT's masked "
            "language model randomly masks 15% of input tokens and predicts "
            "them, allowing the model to learn bidirectional representations "
            "that fuse left and right context."
        ),
        category="pretraining",
    ),
    BenchmarkCase(
        query="How is BERT fine-tuned for downstream tasks?",
        reference_answer=(
            "BERT is fine-tuned by initializing with the pre-trained parameters "
            "and adding a task-specific output layer. All parameters are "
            "fine-tuned end-to-end on labeled data for each task. For token-level "
            "tasks the token representations are fed to an output layer; for "
            "classification tasks the [CLS] representation is used."
        ),
        category="finetuning",
    ),
    BenchmarkCase(
        query="What results did BERT achieve on the GLUE benchmark?",
        reference_answer=(
            "BERT Large achieved a GLUE score of 80.5, an absolute improvement "
            "of 7.7 points over the prior state of the art. It obtained new "
            "state-of-the-art results on all individual GLUE tasks."
        ),
        category="results",
    ),
    BenchmarkCase(
        query="What is the difference between BERT Base and BERT Large?",
        reference_answer=(
            "BERT Base has 12 layers (transformer blocks), 768 hidden size, "
            "12 attention heads, and 110M parameters. BERT Large has 24 layers, "
            "1024 hidden size, 16 attention heads, and 340M parameters."
        ),
        category="architecture",
    ),
]
