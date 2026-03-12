"""
RAG evaluation metrics using Ollama as LLM judge.

Implements faithfulness, answer relevance, context precision,
and ROUGE-L scoring without external evaluation libraries.
"""

import re
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from src.core.config import get_config, Config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class RougeScores:
    """ROUGE-L precision, recall, and F1 scores"""
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


@dataclass
class MetricResult:
    """Result of a single evaluation metric"""
    score: float
    reasoning: str
    error: Optional[str] = None


@dataclass
class EvaluationResult:
    """Aggregated result of all evaluation metrics for a single response"""
    faithfulness: MetricResult
    answer_relevance: MetricResult
    context_precision: MetricResult
    rouge_l: Optional[RougeScores] = None
    evaluation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def aggregate_score(self) -> float:
        """Weighted average of LLM-judge metrics (excludes ROUGE-L)"""
        scores = [
            self.faithfulness.score,
            self.answer_relevance.score,
            self.context_precision.score,
        ]
        return sum(scores) / len(scores)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "faithfulness": {
                "score": self.faithfulness.score,
                "reasoning": self.faithfulness.reasoning,
                "error": self.faithfulness.error,
            },
            "answer_relevance": {
                "score": self.answer_relevance.score,
                "reasoning": self.answer_relevance.reasoning,
                "error": self.answer_relevance.error,
            },
            "context_precision": {
                "score": self.context_precision.score,
                "reasoning": self.context_precision.reasoning,
                "error": self.context_precision.error,
            },
            "aggregate_score": self.aggregate_score,
            "evaluation_time": self.evaluation_time,
            "metadata": self.metadata,
        }
        if self.rouge_l is not None:
            result["rouge_l"] = {
                "precision": self.rouge_l.precision,
                "recall": self.rouge_l.recall,
                "f1": self.rouge_l.f1,
            }
        return result


def _lcs_length(x: List[str], y: List[str]) -> int:
    """Compute length of the longest common subsequence between two token lists."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def compute_rouge_l(generated: str, reference: str) -> RougeScores:
    """
    Compute ROUGE-L scores between generated and reference text.

    Uses longest common subsequence at the word level.
    """
    gen_tokens = generated.lower().split()
    ref_tokens = reference.lower().split()

    if not gen_tokens or not ref_tokens:
        return RougeScores()

    lcs_len = _lcs_length(gen_tokens, ref_tokens)

    precision = lcs_len / len(gen_tokens) if gen_tokens else 0.0
    recall = lcs_len / len(ref_tokens) if ref_tokens else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    return RougeScores(precision=precision, recall=recall, f1=f1)


def _parse_score(text: str, max_value: int = 10) -> Optional[float]:
    """Extract a numeric score from LLM output. Returns normalized 0-1 value."""
    patterns = [
        rf"Score:\s*(\d+(?:\.\d+)?)\s*/\s*{max_value}",
        rf"Score:\s*(\d+(?:\.\d+)?)",
        rf"(\d+(?:\.\d+)?)\s*/\s*{max_value}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            raw = float(match.group(1))
            normalized = min(raw / max_value, 1.0)
            return max(normalized, 0.0)
    return None


class RAGEvaluator:
    """
    Evaluates RAG responses using Ollama as an LLM judge.

    Metrics:
    - Faithfulness: Is the answer grounded in the provided context?
    - Answer Relevance: Does the answer address the question?
    - Context Precision: Are the retrieved chunks relevant to the question?
    - ROUGE-L: Lexical overlap with a reference answer (no LLM needed).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 0.1,
        timeout: Optional[int] = None,
    ):
        if not OLLAMA_AVAILABLE:
            raise ImportError("ollama package not found. Install with: pip install ollama")

        config = None
        if CONFIG_AVAILABLE:
            config = get_config()

        if model is not None:
            self.model = model
        elif config is not None:
            self.model = config.llm.model
        else:
            self.model = "qwen2.5:14b"

        self.temperature = temperature

        if timeout is not None:
            self.timeout = timeout
        elif config is not None:
            self.timeout = config.llm.timeout
        else:
            self.timeout = 120

    def _llm_call(self, prompt: str, max_tokens: int = 500) -> str:
        """Make an Ollama generate call with error handling."""
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": self.temperature,
                    "num_predict": max_tokens,
                },
            )
            return response.get("response", "").strip()
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            raise

    def evaluate_faithfulness(self, context: str, answer: str) -> MetricResult:
        """
        Evaluate whether the answer is grounded in the provided context.

        Asks the LLM to identify claims in the answer and verify each
        against the context. Returns the fraction of supported claims.
        """
        prompt = f"""You are an objective evaluator. Your task is to assess whether an answer is faithfully grounded in the provided context.

Context:
\"\"\"
{context}
\"\"\"

Answer:
\"\"\"
{answer}
\"\"\"

Instructions:
1. List the key factual claims made in the answer.
2. For each claim, determine if it is supported by the context.
3. Count the total claims and the number of supported claims.
4. Provide a score.

Format your response exactly as:
Claims: [list each claim on its own line, marking SUPPORTED or UNSUPPORTED]
Supported: [number of supported claims]
Total: [number of total claims]
Score: [supported/total as a value 0-10, where 10 means all claims are supported]
Reasoning: [brief explanation]"""

        try:
            output = self._llm_call(prompt, max_tokens=600)
            score = _parse_score(output)
            if score is None:
                return MetricResult(
                    score=0.0,
                    reasoning=output,
                    error="Could not parse score from LLM output",
                )
            return MetricResult(score=score, reasoning=output)
        except Exception as e:
            return MetricResult(
                score=0.0,
                reasoning="",
                error=f"Faithfulness evaluation failed: {e}",
            )

    def evaluate_answer_relevance(self, question: str, answer: str) -> MetricResult:
        """
        Evaluate how well the answer addresses the question.

        Asks the LLM to rate the answer's relevance to the question.
        """
        prompt = f"""You are an objective evaluator. Your task is to assess how well an answer addresses the given question.

Question:
\"\"\"
{question}
\"\"\"

Answer:
\"\"\"
{answer}
\"\"\"

Instructions:
1. Determine if the answer directly addresses the question.
2. Check if the answer is complete and covers the main aspects of the question.
3. Assess if the answer stays on topic without unnecessary information.
4. Provide a score.

Format your response exactly as:
Score: [0-10, where 10 means the answer perfectly addresses the question]
Reasoning: [brief explanation of why you gave this score]"""

        try:
            output = self._llm_call(prompt, max_tokens=400)
            score = _parse_score(output)
            if score is None:
                return MetricResult(
                    score=0.0,
                    reasoning=output,
                    error="Could not parse score from LLM output",
                )
            return MetricResult(score=score, reasoning=output)
        except Exception as e:
            return MetricResult(
                score=0.0,
                reasoning="",
                error=f"Answer relevance evaluation failed: {e}",
            )

    def evaluate_context_precision(
        self, question: str, chunks: List[str]
    ) -> MetricResult:
        """
        Evaluate whether the retrieved chunks are relevant to the question.

        Checks each chunk individually and returns the fraction that are relevant.
        """
        if not chunks:
            return MetricResult(
                score=0.0,
                reasoning="No chunks provided",
                error="Empty chunk list",
            )

        relevant_count = 0
        chunk_assessments = []

        for i, chunk in enumerate(chunks):
            prompt = f"""You are an objective evaluator. Determine if the following text chunk is relevant to answering the question.

Question:
\"\"\"
{question}
\"\"\"

Text chunk:
\"\"\"
{chunk}
\"\"\"

Is this chunk relevant to answering the question? A chunk is relevant if it contains information that could help answer the question, even partially.

Format your response exactly as:
Relevant: [YES or NO]
Reason: [one sentence explanation]"""

            try:
                output = self._llm_call(prompt, max_tokens=100)
                is_relevant = bool(re.search(r"Relevant:\s*YES", output, re.IGNORECASE))
                if is_relevant:
                    relevant_count += 1
                chunk_assessments.append(
                    f"Chunk {i + 1}: {'RELEVANT' if is_relevant else 'NOT RELEVANT'}"
                )
            except Exception as e:
                chunk_assessments.append(f"Chunk {i + 1}: ERROR ({e})")

        score = relevant_count / len(chunks)
        reasoning = (
            f"{relevant_count}/{len(chunks)} chunks relevant. "
            + "; ".join(chunk_assessments)
        )

        return MetricResult(score=score, reasoning=reasoning)

    def evaluate_response(
        self,
        question: str,
        answer: str,
        context: str,
        chunks: Optional[List[str]] = None,
        reference_answer: Optional[str] = None,
    ) -> EvaluationResult:
        """
        Run all evaluation metrics on a single RAG response.

        Args:
            question: The user's query.
            answer: The generated answer.
            context: The combined context string passed to the LLM.
            chunks: Individual retrieved text chunks (for context precision).
                    If not provided, context is split on double newlines.
            reference_answer: Optional ground truth answer for ROUGE-L.

        Returns:
            EvaluationResult with all metric scores.
        """
        start_time = time.time()

        if chunks is None:
            chunks = [c.strip() for c in context.split("\n\n") if c.strip()]

        faithfulness = self.evaluate_faithfulness(context, answer)
        answer_relevance = self.evaluate_answer_relevance(question, answer)
        context_precision = self.evaluate_context_precision(question, chunks)

        rouge_l = None
        if reference_answer:
            rouge_l = compute_rouge_l(answer, reference_answer)

        elapsed = time.time() - start_time

        return EvaluationResult(
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_precision=context_precision,
            rouge_l=rouge_l,
            evaluation_time=elapsed,
        )
