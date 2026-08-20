"""Shared helpers for working with VisualCite table JSON.

These helpers are intentionally dependency-free so they can be used from both:
- script-style execution (e.g., `cd src; python benchmark_runner.py`)
- package-style execution (e.g., `python -m src.benchmark_runner`)
"""

from typing import Any, Dict, List


def get_all_table_cells(table_json: Dict[str, Any]) -> List[str]:
    """Extract all cell coordinates from a table JSON.

    Returns Excel-like coordinates (A1, B2, ...), row-major.
    """
    texts = table_json.get("texts", []) if isinstance(table_json, dict) else []
    if not texts:
        return []

    cells: List[str] = []
    for row_idx, row in enumerate(texts):
        for col_idx, _ in enumerate(row):
            col_letter = ""
            col = col_idx
            while col >= 0:
                col_letter = chr(ord("A") + (col % 26)) + col_letter
                col = col // 26 - 1
                if col < 0:
                    break
            row_num = row_idx + 1
            cells.append(f"{col_letter}{row_num}")

    return cells


def get_cell_values(table_json: Dict[str, Any]) -> Dict[str, str]:
    """Get mapping of cell coordinates to their values from a table JSON."""
    texts = table_json.get("texts", []) if isinstance(table_json, dict) else []
    cell_values: Dict[str, str] = {}

    for row_idx, row in enumerate(texts):
        for col_idx, value in enumerate(row):
            col_letter = ""
            col = col_idx
            while col >= 0:
                col_letter = chr(ord("A") + (col % 26)) + col_letter
                col = col // 26 - 1
                if col < 0:
                    break
            row_num = row_idx + 1
            cell_coord = f"{col_letter}{row_num}"
            cell_values[cell_coord] = str(value) if value is not None else ""

    return cell_values
