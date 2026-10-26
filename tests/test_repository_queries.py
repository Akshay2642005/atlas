"""Lexical retrieval must OR a question's terms, not AND them."""

from __future__ import annotations

from sqlalchemy.dialects import postgresql


def _compiled(text: str):
    from atlas_vector.repositories import _lexical_tsquery

    return _lexical_tsquery(text).compile(dialect=postgresql.dialect())


def test_lexical_query_ors_the_question_terms() -> None:
    compiled = _compiled("What vector dimension is assigned to each chunk?")
    sql = str(compiled)

    # `plainto_tsquery` ANDs every lexeme, which matches nothing for a
    # question-shaped query. The OR form is the regression guard.
    assert "to_tsquery" in sql
    assert "array_to_string" in sql
    assert "plainto_tsquery" not in sql
    assert " | " in compiled.params.values()


def test_lexical_query_guards_a_stopword_only_input() -> None:
    # to_tsquery("english", "") raises, so the empty case needs a fallback.
    compiled = _compiled("the and of")

    assert "nullif" in str(compiled)
    assert "atlasempty" in compiled.params.values()
