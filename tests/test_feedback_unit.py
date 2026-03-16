"""Unit tests — run without an API key using mocked LLM responses."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.feedback import get_feedback, _cache, _cache_key
from app.models import FeedbackRequest


def _mock_anthropic_response(response_data: dict) -> MagicMock:
    """Build a mock Anthropic Messages response."""
    content_block = MagicMock()
    content_block.text = json.dumps(response_data)
    msg = MagicMock()
    msg.content = [content_block]
    return msg


def _make_request(sentence, target="Spanish", native="English"):
    return FeedbackRequest(
        sentence=sentence,
        target_language=target,
        native_language=native,
    )


@pytest.fixture(autouse=True)
def clear_cache():
    """Ensure each test starts with an empty cache."""
    _cache.clear()
    yield
    _cache.clear()


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feedback_with_single_error():
    mock_response = {
        "corrected_sentence": "Yo fui al mercado ayer.",
        "is_correct": False,
        "errors": [
            {
                "original": "soy fue",
                "correction": "fui",
                "error_type": "conjugation",
                "explanation": "You mixed two verb forms.",
            }
        ],
        "difficulty": "A2",
    }

    with patch("app.feedback.anthropic.AsyncAnthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(mock_response)
        )
        result = await get_feedback(
            _make_request("Yo soy fue al mercado ayer.", "Spanish", "English")
        )

    assert result.is_correct is False
    assert result.corrected_sentence == "Yo fui al mercado ayer."
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "conjugation"
    assert result.difficulty == "A2"


@pytest.mark.asyncio
async def test_feedback_correct_sentence():
    sentence = "Ich habe gestern einen interessanten Film gesehen."
    mock_response = {
        "corrected_sentence": sentence,
        "is_correct": True,
        "errors": [],
        "difficulty": "B1",
    }

    with patch("app.feedback.anthropic.AsyncAnthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(mock_response)
        )
        result = await get_feedback(_make_request(sentence, "German", "English"))

    assert result.is_correct is True
    assert result.errors == []
    assert result.corrected_sentence == sentence
    assert result.difficulty == "B1"


@pytest.mark.asyncio
async def test_feedback_multiple_errors():
    mock_response = {
        "corrected_sentence": "Le chat noir est sur la table.",
        "is_correct": False,
        "errors": [
            {
                "original": "La chat",
                "correction": "Le chat",
                "error_type": "gender_agreement",
                "explanation": "'Chat' is masculine.",
            },
            {
                "original": "le table",
                "correction": "la table",
                "error_type": "gender_agreement",
                "explanation": "'Table' is feminine.",
            },
        ],
        "difficulty": "A1",
    }

    with patch("app.feedback.anthropic.AsyncAnthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(mock_response)
        )
        result = await get_feedback(
            _make_request("La chat noir est sur le table.", "French", "English")
        )

    assert result.is_correct is False
    assert len(result.errors) == 2
    assert all(e.error_type == "gender_agreement" for e in result.errors)


@pytest.mark.asyncio
async def test_non_latin_script_japanese():
    mock_response = {
        "corrected_sentence": "私は東京に住んでいます。",
        "is_correct": False,
        "errors": [
            {
                "original": "を",
                "correction": "に",
                "error_type": "grammar",
                "explanation": "The verb 住む takes に for location, not を.",
            }
        ],
        "difficulty": "A2",
    }

    with patch("app.feedback.anthropic.AsyncAnthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(mock_response)
        )
        result = await get_feedback(
            _make_request("私は東京を住んでいます。", "Japanese", "English")
        )

    assert result.is_correct is False
    assert any("に" in e.correction for e in result.errors)


@pytest.mark.asyncio
async def test_consistency_correction_forces_is_correct_false():
    """If errors list is non-empty, is_correct must be coerced to False."""
    # Model inconsistently returns is_correct=True with errors present
    mock_response = {
        "corrected_sentence": "Yo fui al mercado.",
        "is_correct": True,  # wrong — should be corrected by our code
        "errors": [
            {
                "original": "soy",
                "correction": "fui",
                "error_type": "conjugation",
                "explanation": "Wrong form.",
            }
        ],
        "difficulty": "A2",
    }

    with patch("app.feedback.anthropic.AsyncAnthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(mock_response)
        )
        result = await get_feedback(_make_request("Yo soy al mercado."))

    # Our consistency check should have fixed this
    assert result.is_correct is False


@pytest.mark.asyncio
async def test_correct_sentence_returns_original_unchanged():
    """For a correct sentence, corrected_sentence must equal the original."""
    sentence = "Je mange une pomme."
    mock_response = {
        "corrected_sentence": "Je mange une pomme.",
        "is_correct": True,
        "errors": [],
        "difficulty": "A1",
    }

    with patch("app.feedback.anthropic.AsyncAnthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(mock_response)
        )
        result = await get_feedback(_make_request(sentence, "French", "English"))

    assert result.corrected_sentence == sentence


@pytest.mark.asyncio
async def test_cache_hit_skips_api_call():
    """Identical requests must be served from cache without a second API call."""
    sentence = "Elle est belle."
    mock_response = {
        "corrected_sentence": sentence,
        "is_correct": True,
        "errors": [],
        "difficulty": "A1",
    }

    with patch("app.feedback.anthropic.AsyncAnthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(mock_response)
        )

        req = _make_request(sentence, "French", "English")
        await get_feedback(req)
        await get_feedback(req)  # second call — should hit cache

        assert instance.messages.create.call_count == 1


@pytest.mark.asyncio
async def test_native_language_preserved_in_explanation():
    """Explanation field should contain native-language text (smoke check)."""
    mock_response = {
        "corrected_sentence": "Ich gehe zur Schule.",
        "is_correct": False,
        "errors": [
            {
                "original": "gehe zu",
                "correction": "gehe zur",
                "error_type": "grammar",
                "explanation": "Du brauchst den Dativ hier: 'zur Schule'.",
            }
        ],
        "difficulty": "A2",
    }

    with patch("app.feedback.anthropic.AsyncAnthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(mock_response)
        )
        result = await get_feedback(
            _make_request("Ich gehe zu Schule.", "German", "German")
        )

    assert len(result.errors[0].explanation) > 0


@pytest.mark.asyncio
async def test_valid_difficulty_enum():
    """Difficulty must be one of the CEFR levels."""
    valid = {"A1", "A2", "B1", "B2", "C1", "C2"}
    mock_response = {
        "corrected_sentence": "Hola.",
        "is_correct": True,
        "errors": [],
        "difficulty": "A1",
    }

    with patch("app.feedback.anthropic.AsyncAnthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(mock_response)
        )
        result = await get_feedback(_make_request("Hola.", "Spanish", "English"))

    assert result.difficulty in valid


@pytest.mark.asyncio
async def test_valid_error_type_enum():
    """Error types must be from the allowed set."""
    valid_types = {
        "grammar", "spelling", "word_choice", "punctuation", "word_order",
        "missing_word", "extra_word", "conjugation", "gender_agreement",
        "number_agreement", "tone_register", "other",
    }
    mock_response = {
        "corrected_sentence": "Eu quero um presente.",
        "is_correct": False,
        "errors": [
            {
                "original": "prezente",
                "correction": "presente",
                "error_type": "spelling",
                "explanation": "The word 'present/gift' is spelled 'presente' in Portuguese.",
            }
        ],
        "difficulty": "A2",
    }

    with patch("app.feedback.anthropic.AsyncAnthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(mock_response)
        )
        result = await get_feedback(
            _make_request("Eu quero um prezente.", "Portuguese", "English")
        )

    for error in result.errors:
        assert error.error_type in valid_types