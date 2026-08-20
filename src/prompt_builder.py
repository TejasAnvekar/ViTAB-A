"""
Prompt builder for VisualCite attribution task
"""
import logging
from typing import Any, Optional, List
from config import PromptStrategy, DataRepresentation, PROMPT_TEMPLATES, COT_EXTRACTION_SUFFIX
from data_loader import VisualCiteSample, VisualCiteDataset, extract_cells_from_formulas, decode_image

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Builds prompts for cell attribution task"""

    def __init__(self, jsonl_path: str = "../visualcite.jsonl", num_examples: int = 10):
        """
        Initialize prompt builder.

        Args:
            jsonl_path: Path to JSONL dataset file
            num_examples: Number of few-shot examples to load (default: 2)
        """
        self.jsonl_path = jsonl_path
        self.num_examples = num_examples
        self.few_shot_examples: Optional[List[VisualCiteSample]] = None
        self._load_few_shot_examples()

    def _load_few_shot_examples(self) -> None:
        """Load few-shot examples from validation split"""
        try:
            dataset = VisualCiteDataset(
                jsonl_path=self.jsonl_path,
                split="validation",
                max_samples=self.num_examples
            )
            dataset.load()

            if len(dataset) > 0:
                self.few_shot_examples = dataset.samples[:self.num_examples]
                logger.info(f"Loaded {len(self.few_shot_examples)} few-shot examples from validation split")
            else:
                logger.warning("No validation samples found, few-shot prompting will use placeholders")
                self.few_shot_examples = None
        except Exception as e:
            logger.warning(f"Failed to load few-shot examples: {e}. Will use placeholders.")
            self.few_shot_examples = None

    def build_prompt(
        self,
        sample: VisualCiteSample,
        strategy: PromptStrategy,
        representation: DataRepresentation,
        table_content: Any
    ) -> tuple[str, Optional[Any]]:
        """
        Build prompt for a given sample and strategy.

        Args:
            sample: VisualCite sample
            strategy: Prompting strategy (zero-shot, few-shot, CoT)
            representation: Data representation type
            table_content: The table content (string for JSON/MD, Image for image)

        Returns:
            Tuple of (formatted_prompt, example_image)
            - formatted_prompt: The text prompt
            - example_image: PIL Image for few-shot example (None for zero-shot/CoT or text representations)
        """
        # Get base template
        template = PROMPT_TEMPLATES[strategy]

        # Prepare table string
        if isinstance(table_content, str):
            table_str = table_content
        else:
            # For images, we don't include table in text prompt
            table_str = "[TABLE IMAGE PROVIDED]"

        # Format based on strategy
        if strategy == PromptStrategy.ZERO_SHOT:
            prompt = template.format(
                table=table_str,
                question=sample.question,
                answer=sample.answer
            )
            return prompt, None

        elif strategy == PromptStrategy.FEW_SHOT:
            example_image = None
            
            # Use real examples from validation split if available
            if self.few_shot_examples and len(self.few_shot_examples) >= 2:
                # Get table representation for example (matching the current representation)
                ex1 = self.few_shot_examples[1]  # Use second example (index 1)

                # Get table content based on representation - match the current representation
                if isinstance(table_content, str):
                    # For text representations (JSON/Markdown), use the same representation
                    if representation == DataRepresentation.MARKDOWN:
                        ex1_table = ex1.table_md
                    elif representation == DataRepresentation.JSON:
                        ex1_table = str(ex1.table_json)
                    else:
                        ex1_table = ex1.table_md  # Default fallback
                else:
                    # For image representations, load the matching style image for the example
                    # Extract style from representation (e.g., image_arial -> arial)
                    style = representation.value.replace("image_", "")
                    if style == "times":
                        style = "times_new_roman"
                    
                    # Load the example image in the same style
                    if style in ex1.table_images:
                        example_image = decode_image(ex1.table_images[style])
                        logger.info(f"Few-shot example image: style={style}, size={example_image.size if example_image else None}, sample_id={ex1.id if hasattr(ex1, 'id') else 'unknown'}, source_id={ex1.source_id if hasattr(ex1, 'source_id') else 'unknown'}")
                    else:
                        # Fallback to arial if specific style not available
                        logger.warning(f"Example doesn't have '{style}' style, using 'arial'")
                        example_image = decode_image(ex1.table_images.get('arial', ''))
                        logger.info(f"Few-shot example image: style=arial (fallback), size={example_image.size if example_image else None}, sample_id={ex1.id if hasattr(ex1, 'id') else 'unknown'}, source_id={ex1.source_id if hasattr(ex1, 'source_id') else 'unknown'}")
                    
                    # For images: Don't include "TABLE:" line, image will be positioned directly
                    ex1_table = "<IMAGE>"

                # Format ground truth cells from formulas
                ex1_cells = ", ".join(extract_cells_from_formulas(ex1.answer_formulas)) if ex1.answer_formulas else "N/A"

                # Add "=" prefix to match formula format
                if ex1_cells != "N/A" and not ex1_cells.startswith("="):
                    ex1_cells = "=" + ex1_cells

                # For images, modify the template to work with image positioning
                if isinstance(table_content, str):
                    # Text: use standard template
                    prompt = template.format(
                        example1_table=ex1_table,
                        example1_question=ex1.question,
                        example1_answer=ex1.answer,
                        example1_cells=ex1_cells,
                        table=table_str,
                        question=sample.question,
                        answer=sample.answer
                    )
                else:
                    # Images: create custom structure without "TABLE:" lines
                    prompt = f"""You are a table analysis expert. Your task is to identify which cell(s) in the table contain or support the given answer to the question.

Here is an example:

EXAMPLE:
<IMAGE>
QUESTION: {ex1.question}
ANSWER: {ex1.answer}
ATTRIBUTED CELLS: {ex1_cells}

Now analyze this table:

<IMAGE>
QUESTION: {sample.question}
ANSWER: {sample.answer}

IMPORTANT: Do NOT repeat the example, question, table, or instructions. Output ONLY the cell coordinates in formula format.

ATTRIBUTED CELLS:"""
            else:
                # Fallback to placeholder example if real example not available
                logger.warning("Using placeholder few-shot example (validation example not loaded)")
                prompt = template.format(
                    example1_table="| A | B |\n| 1 | 2 |",
                    example1_question="What is in cell A1?",
                    example1_answer="1",
                    example1_cells="=A1",
                    table=table_str,
                    question=sample.question,
                    answer=sample.answer
                )
            
            return prompt, example_image

        elif strategy == PromptStrategy.CHAIN_OF_THOUGHT:
            prompt = template.format(
                table=table_str,
                question=sample.question,
                answer=sample.answer
            )
            return prompt, None

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def build_cot_extraction_prompt(self, reasoning: str) -> str:
        """
        Build extraction prompt for CoT to get final answer.

        Args:
            reasoning: The model's reasoning text

        Returns:
            Extraction prompt
        """
        return reasoning + COT_EXTRACTION_SUFFIX
