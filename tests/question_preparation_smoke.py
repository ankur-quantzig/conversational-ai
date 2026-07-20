from __future__ import annotations

import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def install_optional_dependency_stubs() -> None:
    if "openai" not in sys.modules:
        openai = types.ModuleType("openai")

        class OpenAI:
            pass

        openai.OpenAI = OpenAI
        sys.modules.setdefault("openai", openai)

    if "azure.ai.documentintelligence" not in sys.modules:
        azure = types.ModuleType("azure")
        azure_ai = types.ModuleType("azure.ai")
        documentintelligence = types.ModuleType("azure.ai.documentintelligence")
        documentintelligence.DocumentIntelligenceClient = object
        azure_core = types.ModuleType("azure.core")
        credentials = types.ModuleType("azure.core.credentials")
        credentials.AzureKeyCredential = object
        sys.modules.setdefault("azure", azure)
        sys.modules.setdefault("azure.ai", azure_ai)
        sys.modules.setdefault("azure.ai.documentintelligence", documentintelligence)
        sys.modules.setdefault("azure.core", azure_core)
        sys.modules.setdefault("azure.core.credentials", credentials)


def main() -> None:
    install_optional_dependency_stubs()

    from app.rag.answer import (
        INSUFFICIENT_EVIDENCE_MESSAGE,
        is_unsupported_answer,
        parse_question_preparation,
        prepare_retrieval_question,
    )

    parsed = parse_question_preparation(
        """
        Some model preface.
        {
          "status": "ready",
          "rephrased_question": "How do AI agents work?",
          "clarification_question": "",
          "issue": "grammar",
          "confidence_score": 0.91,
          "reason": "The intent is clear, but the wording needed correction."
        }
        """
    )
    assert parsed.status == "ready"
    assert parsed.issue == "grammar"
    assert parsed.rephrased_question == "How do AI agents work?"

    vague = prepare_retrieval_question("?")
    assert vague.status == "needs_clarification"
    assert vague.issue == "vague"
    assert vague.clarification_question.endswith("?")

    assert is_unsupported_answer(INSUFFICIENT_EVIDENCE_MESSAGE)

    print("question preparation smoke ok")


if __name__ == "__main__":
    main()
