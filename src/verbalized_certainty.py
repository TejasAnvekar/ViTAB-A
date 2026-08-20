"""
Verbalized Certainty Extraction through Confidence Querying Prompts.

Implements the Confidence Querying Prompt (CQP) methodology from the paper
"Confidence Under the Hood" (ACL 2024), Section 2.3.

The CQP design includes three key components:
1. Third-Person Perspective (TPP): Presents the Q&A as from another model
2. Option Contextualization (OC): Provides all options for comparative evaluation
3. Likert Scale Utilization (LSU): Uses qualitative certainty scale

For the VisualCite cell attribution task, we adapt this to:
- Present the table, question, answer, and predicted cells
- Ask the model to evaluate certainty in the cell selection
"""
import logging
import re
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

try:
    # Package import (e.g., `python -m src.benchmark_runner`)
    from .confidence_types import (
        CertaintyLevel,
        VerbalizedCertaintyResult,
    )
except ImportError:  # pragma: no cover
    # Script import (e.g., `cd src; python benchmark_runner.py`)
    from confidence_types import (
        CertaintyLevel,
        VerbalizedCertaintyResult,
    )

logger = logging.getLogger(__name__)


def format_cells_by_row(all_cells: List[str]) -> str:
    """
    Format cell coordinates grouped by row for readability.

    Args:
        all_cells: List of cell coordinates (e.g., ["A1", "B1", "A2", "B2", ...])

    Returns:
        Formatted string with cells grouped by row
        e.g., "Row 1: A1, B1, C1 | Row 2: A2, B2, C2 | ..."
    """
    if not all_cells:
        return "N/A"

    # Group cells by row number
    rows_dict: Dict[int, List[str]] = {}
    for cell in all_cells:
        # Extract row number from cell coordinate (e.g., "A1" -> 1, "B12" -> 12)
        match = re.match(r'^([A-Za-z]+)(\d+)$', cell)
        if match:
            row_num = int(match.group(2))
            if row_num not in rows_dict:
                rows_dict[row_num] = []
            rows_dict[row_num].append(cell.upper())

    # Sort rows and format
    sorted_rows = sorted(rows_dict.keys())
    row_strings = []
    for row_num in sorted_rows:
        cells_in_row = sorted(rows_dict[row_num])  # Sort columns alphabetically
        row_strings.append(f"Row {row_num}: {', '.join(cells_in_row)}")

    return " | ".join(row_strings)


# Confidence Querying Prompt Template
# Following the paper's design with TPP, OC, and LSU components

CQP_TEMPLATE_CELL_ATTRIBUTION = """A language model was analyzing a table to identify cells that support an answer.

TABLE:
{table}

QUESTION: {question}
ANSWER: {answer}

All possible cell coordinates in this table: {all_cells_list}

The model selected the following cell(s): {predicted_cells}

Critically analyze whether the identified cells correctly support the answer given all the available cells. Check for missed cells or inclusion of irrelevant cells.
How certain are you that the model's cell selection is completely correct?
a. Very Certain (No doubts at all)
b. Fairly Certain (Minor doubts)
c. Moderately Certain (Some doubts)
d. Somewhat Certain (Significant doubts)
e. Not Certain (Likely incorrect)
f. Very Uncertain (Definitely incorrect)

Answer with just the letter (a-f):"""


class VerbalizedCertaintyExtractor:
    """
    Extracts verbalized certainty by querying the model about its confidence.

    Uses the Confidence Querying Prompt (CQP) methodology from the paper.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        do_sample: bool = False
    ):
        """
        Initialize the verbalized certainty extractor.

        Args:
            model: The language model
            processor: The model's processor/tokenizer
            max_new_tokens: Maximum tokens for certainty response
            temperature: Temperature for generation
            do_sample: Whether to sample during generation
        """
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

    def build_confidence_query_prompt(
        self,
        question: str,
        answer: str,
        predicted_cells: List[str],
        table: Optional[str],
        all_cells: Optional[List[str]] = None
    ) -> str:
        """
        Build the Confidence Querying Prompt (CQP) for a given sample.

        Args:
            question: The original question
            answer: The model's answer
            predicted_cells: Cells predicted by the model
            table: Table representation (markdown or text), or None if using image
            all_cells: All possible cells in the table

        Returns:
            Formatted CQP string
        """
        # Format ALL cells from the table as the option set
        # This is critical for the confidence framework - every cell is a potential choice
        if all_cells:
            # Format cells in a readable grid-like structure by row
            all_cells_list = format_cells_by_row(all_cells)
        else:
            all_cells_list = "N/A"

        # Handle table representation
        if table is not None:
            # Text representation (JSON or Markdown)
            table_text = table
            if len(table_text) > 10000:
                table_text = table_text[:10000] + "\n... (truncated)"
        else:
            # Image representation - table will be passed as image
            table_text = "[TABLE IMAGE PROVIDED]"

        return CQP_TEMPLATE_CELL_ATTRIBUTION.format(
            question=question,
            answer=answer,
            table=table_text,
            predicted_cells=", ".join(predicted_cells),
            all_cells_list=all_cells_list
        )

    def extract_certainty_from_response(self, response: str) -> Tuple[Optional[CertaintyLevel], float, bool]:
        """
        Parse the model's response to extract the certainty level.

        Args:
            response: Raw model response to the CQP

        Returns:
            Tuple of (CertaintyLevel, numerical_score, parse_success)
        """
        response = response.strip()

        # Try to parse using CertaintyLevel enum
        certainty = CertaintyLevel.from_response(response)

        if certainty is not None:
            return certainty, certainty.score, True

        # Try regex patterns for edge cases
        patterns = [
            (r'\ba\.?\s*(?:very certain|verycertain)', CertaintyLevel.VERY_CERTAIN),
            (r'\bb\.?\s*(?:fairly certain|fairlycertain)', CertaintyLevel.FAIRLY_CERTAIN),
            (r'\bc\.?\s*(?:moderately certain|moderatelycertain)', CertaintyLevel.MODERATELY_CERTAIN),
            (r'\bd\.?\s*(?:somewhat certain|somewhatcertain)', CertaintyLevel.SOMEWHAT_CERTAIN),
            (r'\be\.?\s*(?:not certain|notcertain)', CertaintyLevel.NOT_CERTAIN),
            (r'\bf\.?\s*(?:very uncertain|veryuncertain)', CertaintyLevel.VERY_UNCERTAIN),
        ]

        response_lower = response.lower()
        for pattern, level in patterns:
            if re.search(pattern, response_lower):
                return level, level.score, True

        # Check for just the letter at the start
        first_char = response_lower[0] if response_lower else ""
        letter_to_level = {
            'a': CertaintyLevel.VERY_CERTAIN,
            'b': CertaintyLevel.FAIRLY_CERTAIN,
            'c': CertaintyLevel.MODERATELY_CERTAIN,
            'd': CertaintyLevel.SOMEWHAT_CERTAIN,
            'e': CertaintyLevel.NOT_CERTAIN,
            'f': CertaintyLevel.VERY_UNCERTAIN,
        }
        if first_char in letter_to_level:
            level = letter_to_level[first_char]
            return level, level.score, True

        # Try to extract numerical confidence if the model gives a number
        num_match = re.search(r'(\d+(?:\.\d+)?)\s*%', response)
        if num_match:
            percent = float(num_match.group(1))
            score = percent / 100.0
            # Map to closest certainty level
            if score >= 0.9:
                return CertaintyLevel.VERY_CERTAIN, score, True
            elif score >= 0.7:
                return CertaintyLevel.FAIRLY_CERTAIN, score, True
            elif score >= 0.5:
                return CertaintyLevel.MODERATELY_CERTAIN, score, True
            elif score >= 0.3:
                return CertaintyLevel.SOMEWHAT_CERTAIN, score, True
            elif score >= 0.1:
                return CertaintyLevel.NOT_CERTAIN, score, True
            else:
                return CertaintyLevel.VERY_UNCERTAIN, score, True

        # Failed to parse
        logger.warning(f"Failed to parse certainty from response: {response[:100]}")
        return None, 0.5, False  # Default to middle score

    def query_confidence(
        self,
        question: str,
        answer: str,
        predicted_cells: List[str],
        table: Optional[str],
        all_cells: Optional[List[str]] = None,
        image: Optional[Any] = None
    ) -> VerbalizedCertaintyResult:
        """
        Query the model for its verbalized certainty about an answer.

        Args:
            question: The original question
            answer: The model's answer
            predicted_cells: Predicted cells
            table: Table representation (text), or None if using image
            all_cells: All possible cells
            image: Optional image input (for image representations)

        Returns:
            VerbalizedCertaintyResult with the extracted certainty
        """
        import time
        import torch

        # Build the confidence query prompt
        prompt = self.build_confidence_query_prompt(
            question=question,
            answer=answer,
            predicted_cells=predicted_cells,
            table=table,
            all_cells=all_cells
        )

        # Prepare messages
        if image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

        # Generate response
        start_time = time.perf_counter()

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            truncation=False,
            max_length=None
        )
        inputs = inputs.to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.do_sample else None,
                do_sample=self.do_sample,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

        end_time = time.perf_counter()

        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        # Parse the response
        certainty_level, certainty_score, parse_success = self.extract_certainty_from_response(response)

        return VerbalizedCertaintyResult(
            certainty_level=certainty_level,
            certainty_score=certainty_score,
            raw_response=response,
            parse_success=parse_success,
            query_prompt=prompt
        )



