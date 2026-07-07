from __future__ import annotations

from typing import Any


def normalize_table(table) -> dict[str, Any]:
    cells = []
    for cell in table.cells:
        cells.append(
            {
                "row_index": cell.row_index,
                "column_index": cell.column_index,
                "row_span": getattr(cell, "row_span", None),
                "column_span": getattr(cell, "column_span", None),
                "kind": getattr(cell, "kind", None),
                "content": cell.content,
            }
        )
    row_count = getattr(table, "row_count", 0) or 0
    column_count = getattr(table, "column_count", 0) or 0
    grid = [["" for _ in range(column_count)] for _ in range(row_count)]
    for cell in table.cells:
        if 0 <= cell.row_index < row_count and 0 <= cell.column_index < column_count:
            grid[cell.row_index][cell.column_index] = cell.content
    return {
        "page_number": _table_page_number(table),
        "row_count": row_count,
        "column_count": column_count,
        "cells_sample": cells[:20],
        "rows": grid,
        "has_header": bool(getattr(table, "column_count", 0) and any(c.kind == "columnHeader" for c in table.cells)),
        "bounding_regions": [
            {
                "page_number": getattr(region, "page_number", None),
                "polygon": getattr(region, "polygon", None),
            }
            for region in getattr(table, "bounding_regions", []) or []
        ],
    }


def _table_page_number(table) -> int | None:
    for region in getattr(table, "bounding_regions", []) or []:
        page_number = getattr(region, "page_number", None)
        if page_number:
            return page_number
    spans = getattr(table, "spans", []) or []
    if spans:
        return getattr(spans[0], "page_number", None)
    return None

