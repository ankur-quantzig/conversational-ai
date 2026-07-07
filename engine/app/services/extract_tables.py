from __future__ import annotations

import re


TABLE_CAPTION_PATTERN = re.compile(r"^Table\s+\d+\.", re.I)


def reconstruct_tables_from_text(page_text: str) -> list[dict]:
    if not page_text:
        return []

    tables = []
    lines = [line.rstrip() for line in page_text.splitlines()]
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not TABLE_CAPTION_PATTERN.match(line):
            idx += 1
            continue

        caption = line
        idx += 1
        body_lines = []
        while idx < len(lines):
            current = lines[idx].strip()
            if not current:
                if body_lines:
                    idx += 1
                    break
                idx += 1
                continue
            if current.startswith("Table ") and body_lines:
                break
            if current.startswith("Figure ") and body_lines:
                break
            if re.match(r"^\d+\.\s+[A-Z]", current):
                break
            body_lines.append(current)
            idx += 1

        rows = []
        for body_line in body_lines:
            parts = [part.strip() for part in re.split(r"\s{2,}|\t+", body_line) if part.strip()]
            rows.append(parts if len(parts) >= 2 else [body_line])
        if rows:
            max_cols = max(len(row) for row in rows)
            normalized_rows = [row + [""] * (max_cols - len(row)) for row in rows]
            tables.append(
                {
                    "caption": caption,
                    "row_count": len(normalized_rows),
                    "column_count": max_cols,
                    "rows": normalized_rows,
                    "source": "text_fallback",
                    "confidence": "low",
                }
            )
    return tables

