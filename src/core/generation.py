"""
LLM response generation with prompt building and conversation history.
Extracted from ollama_rag.py for modularity.
"""

import time
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def format_conversation_history(conversation_history: list) -> str:
    """Format conversation history for inclusion in prompts.

    Args:
        conversation_history: List of message dicts with "role" and "content" keys

    Returns:
        Formatted conversation string, or empty string if no history
    """
    if not conversation_history:
        return ""

    pairs = []
    i = 0
    while i < len(conversation_history) - 1:
        if (conversation_history[i].get("role") == "user" and
                conversation_history[i + 1].get("role") == "assistant"):
            user_msg = conversation_history[i]["content"]
            assistant_msg = conversation_history[i + 1]["content"][:200]
            pairs.append(f"User: {user_msg}\nAssistant: {assistant_msg}")
            i += 2
        else:
            i += 1

    if not pairs:
        return ""

    recent_pairs = pairs[-5:]
    return "Previous conversation:\n" + "\n\n".join(recent_pairs) + "\n\n"


def build_prompt(query: str, context: str = "", is_summarization: bool = False,
                 require_citations: bool = True, conversation_history: list = None) -> str:
    """Build the appropriate prompt based on query type and context."""
    conv_context = format_conversation_history(conversation_history)

    if context:
        if is_summarization:
            return f"""You are summarizing a document based on the provided context.
Synthesize the information from ALL provided documents into a summary.
Cover the main contributions, methodology, key findings, and conclusions.
Use the format [Document X] when referencing specific information.
Do NOT say you cannot summarize - work with the context provided.

Context:
{context}

{conv_context}Request: {query}

Summary (with citations):"""
        elif require_citations:
            return f"""Answer the following question using the provided context.
IMPORTANT: You must cite specific information from the documents. Use the format [Document X] when referencing information.
Base your answer on the provided context. If the context contains relevant information, synthesize it into a clear answer.
Only say you cannot find the information if the context is truly unrelated to the question.

Context:
{context}

{conv_context}Question: {query}

Answer (with citations):"""
        else:
            return f"""Answer the following question using the provided context. Be accurate and cite information from the context when possible.

Context:
{context}

{conv_context}Question: {query}

Answer:"""
    else:
        return f"""Answer the following question concisely and accurately:

{conv_context}Question: {query}

Answer:"""


def generate_response(client, model: str, prompt: str, temperature: float = 0.3,
                      max_tokens: int = 1000, max_retries: int = 3,
                      verbose: bool = False) -> str:
    """Generate a response from Ollama with retry logic.

    Args:
        client: Ollama client instance
        model: Model name
        prompt: Full prompt string
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate
        max_retries: Number of retry attempts
        verbose: Whether to log detailed info

    Returns:
        Generated response text

    Raises:
        RuntimeError: If all retry attempts fail
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            if verbose:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt + 1}/{max_retries}...")
                else:
                    logger.info(f"Generating response with {model}...")

            start_time = time.time()

            response = client.generate(
                model=model,
                prompt=prompt,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                    'stop': ['Question:', 'Context:']
                }
            )

            generation_time = time.time() - start_time
            answer = response['response'].strip()

            if verbose:
                logger.info(f"Response generated in {generation_time:.1f}s, tokens: {response.get('eval_count', 'N/A')}")

            return answer

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"Ollama API call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Error generating response after {max_retries} attempts: {last_error}")
                raise RuntimeError(f"LLM generation failed after {max_retries} attempts: {last_error}") from last_error
