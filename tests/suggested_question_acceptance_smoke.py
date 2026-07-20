from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from app.api.retrieval import structured_answer_is_usable
    from app.rag.answer import AnswerCitation, RagAnswer, question_has_explicit_subject

    dataset = json.loads(
        (ROOT / "evaluation" / "suggested_questions.v1.json").read_text(encoding="utf-8")
    )
    questions = dataset["questions"]
    assert len(questions) >= 6
    assert all(question_has_explicit_subject(item["question"]) for item in questions)
    assert all(item["expected_topics"] for item in questions)

    citation = AnswerCitation(source_index=1, quote="frame-by-frame OCR")
    grounded = RagAnswer(
        heading="Grounded response",
        answer="- A source-supported response.",
        citations=[citation],
        confidence="medium",
        confidence_score=0.7,
    )
    assert structured_answer_is_usable(grounded)

    grounded.confidence_score = 0.64
    assert not structured_answer_is_usable(grounded)

    grounded.confidence_score = 0.9
    grounded.citations = []
    assert not structured_answer_is_usable(grounded)

    print("suggested question acceptance smoke ok")


if __name__ == "__main__":
    main()
