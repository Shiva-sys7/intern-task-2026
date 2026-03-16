"""LLM interaction for language feedback — powered by Anthropic Claude."""

import asyncio
import hashlib
import json
import logging
import os
from typing import Optional

import anthropic

from app.models import FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert language teacher. A student has written a sentence in their \
target language. Your job is to analyze it, identify every error, and return \
structured JSON feedback.

## Instructions

1. **Analyze carefully.** Check grammar, spelling, vocabulary, punctuation, \
word order, verb conjugation, gender/number agreement, and register.

2. **Correct minimally.** The corrected_sentence should fix all errors but \
preserve the student's vocabulary level, tone, and meaning. Don't rephrase \
what wasn't wrong.

3. **If the sentence is already correct:** set is_correct=true, errors=[], \
and set corrected_sentence to the original sentence unchanged.

4. **CEFR difficulty** reflects the complexity of the sentence itself \
(vocabulary and grammar structures), NOT whether it has errors:
   - A1: Basic isolated phrases, present tense, everyday words
   - A2: Simple connected sentences, past/future, common topics
   - B1: Compound sentences, multiple tenses, abstract topics beginning
   - B2: Complex sentences, passive voice, sustained argument
   - C1: Nuanced expression, idiomatic range, academic/professional
   - C2: Masterful precision, literary structures, native-equivalent

5. **Explanations** must be written in the student's **native language** \
(not the target language). Keep them friendly, 1-2 sentences, focused on \
*why* the correction is needed so the student learns the rule.

6. **Non-Latin scripts** (Japanese, Chinese, Korean, Russian, Arabic, Greek, \
Hebrew, Thai, etc.) are fully supported. Analyze them with the same rigor.

7. **Error types** — use one of these exact strings:
   grammar, spelling, word_choice, punctuation, word_order, missing_word, \
   extra_word, conjugation, gender_agreement, number_agreement, \
   tone_register, other

## Output

Respond with **only** a valid JSON object — no markdown fences, no extra text:

{
  "corrected_sentence": "string",
  "is_correct": boolean,
  "errors": [
    {
      "original": "the erroneous text as it appears in the sentence",
      "correction": "the corrected replacement",
      "error_type": "one of the allowed types above",
      "explanation": "1-2 sentence explanation in the student's native language"
    }
  ],
  "difficulty": "A1 | A2 | B1 | B2 | C1 | C2"
}
"""

# ---------------------------------------------------------------------------
# In-memory response cache (keyed on request content hash)
# ---------------------------------------------------------------------------

_cache: dict[str, FeedbackResponse] = {}


def _cache_key(request: FeedbackRequest) -> str:
    payload = f"{request.sentence}|{request.target_language}|{request.native_language}"
    return hashlib.sha256(payload.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_BASE_DELAY = 1.0  # seconds


async def _call_with_retry(client: anthropic.AsyncAnthropic, user_message: str) -> str:
    """Call the Anthropic API with exponential-backoff retry on transient errors."""
    last_exc: Optional[Exception] = None

    for attempt in range(_MAX_RETRIES):
        try:
            response = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                temperature=0.1,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text
        except anthropic.RateLimitError as exc:
            last_exc = exc
            wait = _BASE_DELAY * (2 ** attempt)
            logger.warning(
                "Rate limit hit (attempt %d/%d); retrying in %.1fs",
                attempt + 1, _MAX_RETRIES, wait,
            )
            await asyncio.sleep(wait)
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                last_exc = exc
                wait = _BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "Server error %d (attempt %d/%d); retrying in %.1fs",
                    exc.status_code, attempt + 1, _MAX_RETRIES, wait,
                )
                await asyncio.sleep(wait)
            else:
                raise

    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def get_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Return structured language feedback for the given learner sentence."""

    # Check cache first — avoids duplicate API calls for identical requests
    key = _cache_key(request)
    if key in _cache:
        logger.debug("Cache hit for request key %s", key[:8])
        return _cache[key]

    client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    user_message = (
        f"Target language: {request.target_language}\n"
        f"Native language: {request.native_language}\n"
        f"Sentence: {request.sentence}"
    )

    raw = await _call_with_retry(client, user_message)

    # Strip accidental markdown fences if the model adds them
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        raw = raw.rsplit("```", 1)[0].strip()

    data = json.loads(raw)

    # Enforce consistency: non-empty errors → is_correct must be False
    if data.get("errors"):
        data["is_correct"] = False
    else:
        data["is_correct"] = True
        data["corrected_sentence"] = request.sentence  # guarantee unchanged

    result = FeedbackResponse(**data)

    # Store in cache
    _cache[key] = result
    return result