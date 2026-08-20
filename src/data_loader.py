"""
Data loading and preprocessing for VisualCite benchmark
"""
import json
import base64
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any, Generator
from dataclasses import dataclass
from PIL import Image
import logging

logger = logging.getLogger(__name__)


@dataclass
class VisualCiteSample:
    """Single sample from VisualCite dataset"""
    id: str
    split: str
    question: str
    answer: str
    answer_formulas: List[str]
    highlighted_cells: List[List[int]]  # [[row, col], ...]
    table_json: Dict[str, Any]
    table_md: str
    table_images: Dict[str, str]  # style -> base64 encoded PNG
    source: str
    source_id: str


def parse_cell_coordinates(highlighted_cells: List[List[int]]) -> List[str]:
    """
    Convert row/col indices to Excel-style coordinates.
    Row indices are 0-based, columns are 0-based.
    Output: A1, B2, etc. (1-indexed rows, A-indexed columns)
    """
    coordinates = []
    for row_idx, col_idx in highlighted_cells:
        # Convert column index to letter (0 -> A, 1 -> B, etc.)
        col_letter = ""
        col = col_idx
        while col >= 0:
            col_letter = chr(ord('A') + (col % 26)) + col_letter
            col = col // 26 - 1
            if col < 0:
                break

        # Row is 1-indexed in Excel notation
        row_num = row_idx + 1
        coordinates.append(f"{col_letter}{row_num}")

    return coordinates


def extract_cells_from_formulas(answer_formulas: List[str]) -> List[str]:
    """
    Extract cell references from Excel-style formulas.

    Examples:
        ["=E7"] -> ["E7"]
        ["SUM(F8)"] -> ["F8"]
        ["=A1+B2"] -> ["A1", "B2"]
        ["SUM(C3:C10)"] -> ["C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"]

    Args:
        answer_formulas: List of formula strings

    Returns:
        List of unique cell coordinates (e.g., ["A1", "B2"])
    """
    import re

    all_cells = []

    for formula in answer_formulas:
        if not formula:
            continue

        # Convert to string if not already (handle cases where formula might be int)
        formula = str(formula).strip()

        # Remove leading '=' if present
        if formula.startswith('='):
            formula = formula[1:].strip()  # Strip again after removing =

        # Pattern to match cell references (A1, AA100, etc.) and ranges (A1:B10)
        # This matches: column letters + row numbers, optionally followed by :column letters + row numbers
        cell_pattern = r'([A-Z]+\d+)(?::([A-Z]+\d+))?'
        matches = re.findall(cell_pattern, formula, re.IGNORECASE)

        for match in matches:
            start_cell = match[0].upper()
            end_cell = match[1].upper() if match[1] else None

            if end_cell:
                # Handle range (e.g., C3:C10)
                expanded_cells = expand_cell_range(start_cell, end_cell)
                all_cells.extend(expanded_cells)
            else:
                # Single cell
                all_cells.append(start_cell)

    # Remove duplicates while preserving order
    seen = set()
    unique_cells = []
    for cell in all_cells:
        if cell not in seen:
            seen.add(cell)
            unique_cells.append(cell)

    return unique_cells


def expand_cell_range(start_cell: str, end_cell: str) -> List[str]:
    """
    Expand a cell range to individual cells.

    Examples:
        expand_cell_range("C3", "C10") -> ["C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"]
        expand_cell_range("A1", "B2") -> ["A1", "A2", "B1", "B2"]

    Args:
        start_cell: Starting cell (e.g., "A1")
        end_cell: Ending cell (e.g., "B2")

    Returns:
        List of all cells in the range
    """
    start_row, start_col = excel_to_indices(start_cell)
    end_row, end_col = excel_to_indices(end_cell)

    if start_row is None or end_row is None:
        return []

    cells = []
    for row in range(start_row, end_row + 1):
        for col in range(start_col, end_col + 1):
            # Convert back to Excel notation
            col_letter = ""
            c = col
            while c >= 0:
                col_letter = chr(ord('A') + (c % 26)) + col_letter
                c = c // 26 - 1
                if c < 0:
                    break
            cells.append(f"{col_letter}{row + 1}")

    return cells


def excel_to_indices(cell_coord: str) -> Tuple[int, int]:
    """
    Convert Excel-style coordinate to 0-indexed row/col.
    E.g., "B3" -> (2, 1) meaning row_idx=2, col_idx=1
    """
    import re
    match = re.match(r'^([A-Za-z]+)(\d+)$', cell_coord.strip())
    if not match:
        return None
    
    col_str, row_str = match.groups()
    
    # Convert column letters to index
    col_idx = 0
    for char in col_str.upper():
        col_idx = col_idx * 26 + (ord(char) - ord('A') + 1)
    col_idx -= 1  # Convert to 0-indexed
    
    # Convert row to 0-indexed
    row_idx = int(row_str) - 1
    
    return (row_idx, col_idx)


def parse_model_output(output: str) -> List[str]:
    """
    Parse model output to extract cell coordinates.
    Handles various formats:
    - Formula style: "=E7", "SUM(F8)", "=A1+B2"
    - Plain coordinates: "A1, B2", "A1 B2", "A1,B2"
    - Mixed formats

    Returns:
        List of cell coordinates (e.g., ["A1", "B2"])
    """
    import re

    # Clean up the output
    output = output.strip()

    # Remove common prefixes/suffixes
    prefixes_to_remove = [
        "ATTRIBUTED CELLS:",
        "attributed cells:",
        "Cells:",
        "cells:",
        "Answer:",
        "answer:",
        "The cells are:",
        "The attributed cells are:",
    ]
    for prefix in prefixes_to_remove:
        if output.lower().startswith(prefix.lower()):
            output = output[len(prefix):].strip()

    # First, try to extract cells from formula format (e.g., "=E7", "SUM(F8)")
    # Check if output looks like a formula
    if '=' in output or any(func in output.upper() for func in ['SUM', 'AVERAGE', 'COUNT', 'MIN', 'MAX']):
        # Treat as formula and extract cells
        cells = extract_cells_from_formulas([output])
        if cells:
            return cells

    # Fallback: Find all cell coordinates using regex
    # Pattern matches: A1, AA1, B23, etc.
    pattern = r'[A-Za-z]+\d+'
    matches = re.findall(pattern, output)

    # Normalize to uppercase
    coordinates = [m.upper() for m in matches]

    # Remove duplicates while preserving order
    seen = set()
    unique_coords = []
    for coord in coordinates:
        if coord not in seen:
            seen.add(coord)
            unique_coords.append(coord)

    return unique_coords


def decode_image(base64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image"""
    img_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(img_data))


class VisualCiteDataset:
    """Dataset handler for VisualCite"""

    def __init__(
        self,
        jsonl_path: str = "../visualcite.jsonl",
        split: str = "test",
        max_samples: Optional[int] = None,
        single_cell_only: bool = True
    ):
        self.jsonl_path = jsonl_path
        self.split = split
        self.max_samples = max_samples
        self.single_cell_only = single_cell_only
        self.samples: List[VisualCiteSample] = []
        self._loaded = False
    
    def load(self) -> None:
        """Load dataset from local JSONL file"""
        import os

        logger.info(f"Loading dataset from {self.jsonl_path}...")

        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(f"Dataset file not found: {self.jsonl_path}")

        # Read JSONL file line by line
        filtered_data = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())

                    # Filter by split
                    if item.get('split', '').lower() == self.split.lower():
                        filtered_data.append(item)

                        # Apply max_samples limit early for efficiency
                        if self.max_samples is not None and len(filtered_data) >= self.max_samples:
                            break

                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing line {line_num}: {e}")
                    continue

        if not filtered_data:
            logger.warning(f"No samples found for split '{self.split}'.")

        # Convert to VisualCiteSample objects
        for item in filtered_data:
            try:
                # Parse answer_formulas - it may be a string representation of a list
                answer_formulas_raw = item.get('answer_formulas', [])
                if isinstance(answer_formulas_raw, str):
                    # Parse string representation of list (e.g., "['=E7']")
                    try:
                        import ast
                        answer_formulas = ast.literal_eval(answer_formulas_raw)
                    except (ValueError, SyntaxError):
                        # If parsing fails, treat as single formula
                        answer_formulas = [answer_formulas_raw] if answer_formulas_raw else []
                else:
                    answer_formulas = answer_formulas_raw if answer_formulas_raw else []

                sample = VisualCiteSample(
                    id=item.get('id', ''),
                    split=item.get('split', self.split),
                    question=item.get('question', ''),
                    answer=str(item.get('answer', '')),
                    answer_formulas=answer_formulas,
                    highlighted_cells=item.get('highlighted_cells', []),
                    table_json=item.get('table_json', {}),
                    table_md=item.get('table_md', ''),
                    table_images=item.get('table_images', {}),
                    source=item.get('source', ''),
                    source_id=item.get('source_id', '')
                )

                # Filter for single-cell answers if requested
                if self.single_cell_only:
                    ground_truth_cells = extract_cells_from_formulas(answer_formulas) if answer_formulas else parse_cell_coordinates(item.get('highlighted_cells', []))
                    if len(ground_truth_cells) != 1:
                        logger.debug(f"Skipping sample {sample.id}: {len(ground_truth_cells)} ground truth cells (need exactly 1)")
                        continue

                self.samples.append(sample)
            except Exception as e:
                logger.warning(f"Error parsing sample {item.get('id', 'unknown')}: {e}")
                continue

        self._loaded = True
        logger.info(f"Loaded {len(self.samples)} samples from {self.jsonl_path}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> VisualCiteSample:
        if not self._loaded:
            self.load()
        return self.samples[idx]
    
    def __iter__(self) -> Generator[VisualCiteSample, None, None]:
        if not self._loaded:
            self.load()
        yield from self.samples
    
    def get_ground_truth_cells(self, sample: VisualCiteSample, use_formulas: bool = True) -> List[str]:
        """
        Get ground truth cell coordinates for a sample.

        Args:
            sample: VisualCiteSample object
            use_formulas: If True, extract cells from answer_formulas.
                         If False, use highlighted_cells (legacy behavior).

        Returns:
            List of cell coordinates (e.g., ["A1", "B2"])
        """
        if use_formulas and sample.answer_formulas:
            return extract_cells_from_formulas(sample.answer_formulas)
        else:
            # Fallback to highlighted_cells if formulas not available or use_formulas=False
            return parse_cell_coordinates(sample.highlighted_cells)
    
    def get_table_representation(
        self,
        sample: VisualCiteSample,
        representation: str
    ) -> Any:
        """
        Get table in specified representation format.
        
        Args:
            sample: VisualCite sample
            representation: One of 'json', 'markdown', 'image_arial', 
                          'image_times_new_roman', 'image_red', 'image_blue', 'image_green'
        
        Returns:
            Table in requested format (str, dict, or PIL.Image)
        """
        if representation == "json":
            return json.dumps(sample.table_json, indent=2)
        elif representation == "markdown":
            return sample.table_md
        elif representation.startswith("image_"):
            style = representation.replace("image_", "")
            if style == "times":
                style = "times_new_roman"
            
            if style in sample.table_images:
                return decode_image(sample.table_images[style])
            else:
                # Fallback to arial if style not available
                logger.warning(f"Style '{style}' not found, using 'arial'")
                return decode_image(sample.table_images.get('arial', ''))
        else:
            raise ValueError(f"Unknown representation: {representation}")


def get_json_table_as_readable(table_json: Dict[str, Any]) -> str:
    """
    Convert table JSON to a more readable text format with coordinates.
    """
    texts = table_json.get('texts', [])
    if not texts:
        return json.dumps(table_json, indent=2)
    
    # Build a grid representation
    lines = []
    
    # Add header with column letters
    num_cols = max(len(row) for row in texts) if texts else 0
    col_headers = [''] + [chr(ord('A') + i) if i < 26 else f"A{chr(ord('A') + i - 26)}" 
                         for i in range(num_cols)]
    lines.append(' | '.join(col_headers))
    lines.append('-' * (len(lines[0]) + 10))
    
    # Add rows with row numbers
    for row_idx, row in enumerate(texts):
        row_num = str(row_idx + 1)
        row_data = [row_num] + [str(cell) if cell else '' for cell in row]
        lines.append(' | '.join(row_data))
    
    return '\n'.join(lines)