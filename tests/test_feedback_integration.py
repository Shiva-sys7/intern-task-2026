"""Integration tests — require ANTHROPIC_API_KEY to be set.

Run with: pytest tests/test_feedback_integration.py -v

These tests make real API calls. Skip them in CI or when no key is available.
"""

import os

import pytest

from app.feedback import get_feedback, _cache
from app.models import FeedbackRequest

pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — skipping integration tests",
)

VALID_ERROR_TYPES = {
    "grammar", "spelling", "word_choice", "punctuation", "word_order",
    "missing_word", "extra_word", "conjugation", "gender_agreement",
    "number_agreement", "tone_register", "other",
}
VALID_DIFFICULTIES = {"A1", "A2", "B1", "B2", "C1", "C2"}


@pytest.fixture(autouse=True)
def clear_cache():
    _cache.clear()
    yield
    _cache.clear()


# ---------------------------------------------------------------------------
# Sentences with errors
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spanish_conjugation_error():
    """Mixed verb forms should be detected as a conjugation error."""
    result = await get_feedback(
        FeedbackRequest(
            sentence="Yo soy fue al mercado ayer.",
            target_language="Spanish",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1
    assert result.difficulty in VALID_DIFFICULTIES
    for e in result.errors:
        assert e.error_type in VALID_ERROR_TYPES
        assert len(e.explanation) > 0


@pytest.mark.asyncio
async def test_french_gender_agreement_errors():
    """Two gender-agreement errors in a single sentence."""
    result = await get_feedback(
        FeedbackRequest(
            sentence="La chat noir est sur le table.",
            target_language="French",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1


@pytest.mark.asyncio
async def test_japanese_particle_error():
    """Incorrect particle を should be corrected to に for 住む."""
    result = await get_feedback(
        FeedbackRequest(
            sentence="私は東京を住んでいます。",
            target_language="Japanese",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert any("に" in e.correction for e in result.errors)


@pytest.mark.asyncio
async def test_portuguese_spelling_error():
    """Spelling error: prezente → presente."""
    result = await get_feedback(
        FeedbackRequest(
            sentence="Eu quero comprar um prezente para minha irmã.",
            target_language="Portuguese",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert any("presente" in e.correction for e in result.errors)


@pytest.mark.asyncio
async def test_korean_particle_error():
    """Korean sentence with wrong object marker."""
    result = await get_feedback(
        FeedbackRequest(
            sentence="나는 학교을 가요.",
            target_language="Korean",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert result.difficulty in VALID_DIFFICULTIES
    for e in result.errors:
        assert e.error_type in VALID_ERROR_TYPES


@pytest.mark.asyncio
async def test_russian_case_error():
    """Russian noun in wrong case after preposition в."""
    result = await get_feedback(
        FeedbackRequest(
            sentence="Я иду в школа.",
            target_language="Russian",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert result.difficulty in VALID_DIFFICULTIES


@pytest.mark.asyncio
async def test_italian_missing_article():
    """Italian sentence missing a required article before mela."""
    result = await get_feedback(
        FeedbackRequest(
            sentence="Ho mangiato mela per colazione.",
            target_language="Italian",
            native_language="English",
        )
    )
    assert result.is_correct is False
    assert result.difficulty in VALID_DIFFICULTIES


# ---------------------------------------------------------------------------
# Correct sentences (no errors)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_correct_german_sentence():
    """A grammatically correct German sentence should have no errors."""
    result = await get_feedback(
        FeedbackRequest(
            sentence="Ich habe gestern einen interessanten Film gesehen.",
            target_language="German",
            native_language="English",
        )
    )
    assert result.is_correct is True
    assert result.errors == []
    assert result.difficulty in VALID_DIFFICULTIES


@pytest.mark.asyncio
async def test_correct_spanish_sentence():
    """A simple correct Spanish sentence."""
    result = await get_feedback(
        FeedbackRequest(
            sentence="Me llamo Carlos y tengo veinte años.",
            target_language="Spanish",
            native_language="English",
        )
    )
    assert result.is_correct is True
    assert result.errors == []
    assert result.corrected_sentence == "Me llamo Carlos y tengo veinte años."


@pytest.mark.asyncio
async def test_correct_japanese_sentence():
    """A correct Japanese sentence should be flagged as correct."""
    result = await get_feedback(
        FeedbackRequest(
            sentence="今日は天気がいいですね。",
            target_language="Japanese",
            native_language="English",
        )
    )
    assert result.is_correct is True
    assert result.errors == []


# ---------------------------------------------------------------------------
# Schema and type invariants
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_error_types_are_valid():
    """Every error returned by the API must use an allowed error_type."""
    result = await get_feedback(
        FeedbackRequest(
            sentence="He go to the store yesterday and buyed many things.",
            target_language="English",
            native_language="Spanish",
        )
    )
    for e in result.errors:
        assert e.error_type in VALID_ERROR_TYPES, f"Unknown error type: {e.error_type}"


@pytest.mark.asyncio
async def test_explanations_non_empty():
    """Each error explanation must be a non-empty string."""
    result = await get_feedback(
        FeedbackRequest(
            sentence="La chat noir est sur le table.",
            target_language="French",
            native_language="English",
        )
    )
    for e in result.errors:
        assert isinstance(e.explanation, str)
        assert len(e.explanation.strip()) > 0


@pytest.mark.asyncio
async def test_native_language_affects_explanation():
    """When native language is Spanish, explanations should be in Spanish."""
    result = await get_feedback(
        FeedbackRequest(
            sentence="I has a cat.",
            target_language="English",
            native_language="Spanish",
        )
    )
    assert result.is_correct is False
    assert len(result.errors) >= 1