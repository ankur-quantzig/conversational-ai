from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from app.rag.prompt_store import (
        PROMPT_DIR,
        REQUIRED_PROMPT_TASKS,
        prompt_text,
        validate_prompt_catalogs,
    )

    validate_prompt_catalogs()

    expected_files = {"system_prompts.json", "human_prompts.json"}
    prompt_files = {path.name for path in PROMPT_DIR.glob("*_prompts.json")}
    assert prompt_files == expected_files, prompt_files

    for role in ("system", "human"):
        catalog = json.loads((PROMPT_DIR / f"{role}_prompts.json").read_text(encoding="utf-8"))
        assert set(catalog) == REQUIRED_PROMPT_TASKS
        for task in REQUIRED_PROMPT_TASKS:
            rendered = prompt_text(
                role,
                task,
                question="What changed?",
                context="Source evidence.",
                schema='{"answer": "string"}',
                answer="- A supported fact.",
                history="Previous conversation.",
                source_metadata="document.pdf",
                glossary="RAG: retrieval-augmented generation",
                heuristic="No warnings",
                raw_content="Raw content",
            )
            assert rendered
            assert "DO:" in rendered
            assert "DO NOT:" in rendered

    api_tree = ast.parse(
        (ROOT / "backend" / "app" / "api" / "main.py").read_text(encoding="utf-8")
    )
    progress_assignment = next(
        node
        for node in api_tree.body
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "USER_PROGRESS"
        )
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "USER_PROGRESS"
                for target in node.targets
            )
        )
    )
    progress_mapping = ast.literal_eval(progress_assignment.value)
    public_text = " ".join(
        f"{stage} {message}" for stage, message in progress_mapping.values()
    ).lower()
    for internal_term in (
        "bm25",
        "vector search",
        "chunks",
        "reranking",
        "adding helpful next questions",
    ):
        assert internal_term not in public_text, internal_term

    print("Prompt catalogs and user-facing progress labels validated.")


if __name__ == "__main__":
    main()
