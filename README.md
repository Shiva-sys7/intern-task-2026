# Language Feedback API

An LLM-powered REST API that analyzes language-learner sentences and returns
structured, educational correction feedback. Built with **FastAPI** and
**Anthropic Claude** (`claude-haiku-4-5`).

---

## Quick Start

```bash
# 1. Copy env file and add your Anthropic API key
cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY=sk-ant-...

# 2. Start with Docker
docker compose up --build

# 3. Test the health endpoint
curl http://localhost:8000/health

# 4. Send a feedback request
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "sentence": "Yo soy fue al mercado ayer.",
    "target_language": "Spanish",
    "native_language": "English"
  }'
```

---

## Running Tests

```bash
# Unit tests (no API key required — all LLM calls are mocked)
pytest tests/test_feedback_unit.py tests/test_schema.py -v

# Integration tests (requires ANTHROPIC_API_KEY in .env)
pytest tests/test_feedback_integration.py -v

# All tests
pytest -v
```

---

## Design Decisions

### Model: `claude-haiku-4-5`

I chose Claude Haiku for three reasons:

1. **Speed** — typical response time is 2–5 seconds, well under the 30-second
   limit. Using a larger model would gain marginal accuracy at the cost of
   latency and significant extra expense.
2. **Cost** — language feedback is a high-volume operation. At production
   scale, the cost difference between Haiku and Sonnet/Opus is substantial.
3. **Accuracy** — for structured grammar analysis with a tight system prompt
   and low temperature (0.1), Haiku performs as well as much larger models.
   Language correction is pattern-matching, not deep reasoning.

### Prompt Engineering

The system prompt is written in three parts:

1. **Role** — frames Claude as a language teacher, not a chatbot, setting
   the right behavioral mode.
2. **Detailed rules** — each requirement is stated explicitly:
   - minimal corrections (preserve student voice)
   - is_correct/errors consistency
   - CEFR rating based on sentence complexity, not error presence
   - explanations in the native language, not the target language
   - explicit CEFR level definitions so ratings are consistent
   - non-Latin script support reminder
3. **Output contract** — the exact JSON schema is given inline. This makes
   the output format unambiguous and easy to validate.

I use `temperature=0.1` to reduce variation in error classification and
difficulty ratings. Language feedback should be deterministic.

### Consistency Enforcement

After parsing the LLM response, the code enforces two invariants:

- If `errors` is non-empty → `is_correct` is coerced to `False`
- If `errors` is empty → `is_correct` is coerced to `True` and
  `corrected_sentence` is set to the original

This handles the rare case where the model's response is internally
inconsistent.

### In-Memory Cache

Identical requests (same sentence + language pair) are served from a
dict-based cache without hitting the API again. This is important for:
- Reducing cost in production (students often retry the same sentence)
- Improving response time on repeated requests

For a real production deployment, this would be replaced with Redis (keyed
on the SHA-256 hash of the request), with a short TTL (e.g., 1 hour).

### Retry Logic

Transient API errors (rate limits, 5xx server errors) are retried up to
3 times with exponential backoff (1s, 2s, 4s). This makes the service
resilient to brief Anthropic API hiccups without failing the user's request.

### Test Strategy

Tests are split into three layers:

| File | What it tests | Needs API key? |
|------|---------------|----------------|
| `test_schema.py` | JSON schema validation, Pydantic models | No |
| `test_feedback_unit.py` | Business logic, caching, consistency enforcement | No (mocked) |
| `test_feedback_integration.py` | Real LLM accuracy across 8+ languages and edge cases | Yes |

The unit tests mock the Anthropic client at the module level, so they run
instantly and never touch the network. Integration tests cover:
- Spanish (conjugation error)
- French (gender agreement, two errors)
- Japanese (particle error, non-Latin script)
- Portuguese (spelling error)
- Korean (particle error, non-Latin script)
- Russian (case error, Cyrillic)
- Italian (missing article)
- German (correct sentence — no false positives)
- English (multiple errors, non-English native speaker)

### Production Considerations

If this were deployed to production for Pangea Chat, I would add:

- **Redis cache** with TTL instead of in-process dict
- **Request validation middleware** (sentence length limit, language allowlist)
- **Structured logging** with request IDs for tracing
- **Cost monitoring** — log token counts per request to track spend
- **Circuit breaker** around the Anthropic client for graceful degradation

---

## API Reference

### `GET /health`

Returns `{"status": "ok"}` with HTTP 200. Used by Docker healthcheck.

### `POST /feedback`

**Request body:**

```json
{
  "sentence": "string (min length 1)",
  "target_language": "string (min length 2)",
  "native_language": "string (min length 2)"
}
```

**Response body:**

```json
{
  "corrected_sentence": "string",
  "is_correct": boolean,
  "errors": [
    {
      "original": "string",
      "correction": "string",
      "error_type": "grammar | spelling | word_choice | punctuation | word_order | missing_word | extra_word | conjugation | gender_agreement | number_agreement | tone_register | other",
      "explanation": "string (written in the native language)"
    }
  ],
  "difficulty": "A1 | A2 | B1 | B2 | C1 | C2"
}
```

---

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app, /health, /feedback endpoints
│   ├── models.py        # Pydantic request/response models
│   └── feedback.py      # LLM prompt, Anthropic client, cache, retry
├── schema/
│   ├── request.schema.json
│   └── response.schema.json
├── tests/
│   ├── test_schema.py              # Schema + model validation (no API needed)
│   ├── test_feedback_unit.py       # Mocked unit tests (no API needed)
│   └── test_feedback_integration.py # Real LLM tests (API key required)
├── examples/
│   └── sample_inputs.json
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── pytest.ini
```