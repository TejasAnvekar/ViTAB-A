import ast
import json
import base64
import random
import asyncio
import argparse
from io import BytesIO
from pathlib import Path
from tqdm.asyncio import tqdm
from datasets import load_dataset
from playwright.async_api import async_playwright


def parse_table_content(table_content_str):
    return ast.literal_eval(table_content_str)


def column_letter(col_idx):
    letter = ''
    col_idx += 1
    while col_idx > 0:
        col_idx -= 1
        letter = chr(col_idx % 26 + ord('A')) + letter
        col_idx //= 26
    return letter


def table_to_markdown(table_dict):
    '''
    Markdown tables don't support merged cells, so merged regions
    will display the value only in the first cell of the merged region.
    '''
    texts = table_dict['texts']
    merged_regions = table_dict['merged_regions']
    top_header_rows = table_dict['top_header_rows_num']
    title = table_dict.get('title', '')

    num_rows = len(texts)
    num_cols = len(texts[0]) if texts else 0

    # Track which cells should be skipped due to merging
    skip_cells = set()

    for region in merged_regions:
        first_row, last_row = region['first_row'], region['last_row']
        first_col, last_col = region['first_column'], region['last_column']

        # Mark all cells except the first one in the merged region as skip
        for r in range(first_row, last_row + 1):
            for c in range(first_col, last_col + 1):
                if r != first_row or c != first_col:
                    skip_cells.add((r, c))

    markdown = ""

    # Row counter starts at 1
    current_row = 1

    # Column header row (row number + A, B, C, ...)
    col_labels = [str(current_row)] + [column_letter(i) for i in range(num_cols)]
    markdown += "| " + " | ".join(col_labels) + " |\n"
    current_row += 1

    # Add separator after column labels
    markdown += "| " + " | ".join(["---"] * (num_cols + 1)) + " |\n"

    # Title row if present (spans all columns)
    if title:
        title_row = [str(current_row), title] + [""] * (num_cols - 1)
        markdown += "| " + " | ".join(title_row) + " |\n"
        current_row += 1

    # Build table content rows with row numbers continuing
    for row_idx in range(num_rows):
        row_cells = [str(current_row)]
        current_row += 1

        for col_idx in range(num_cols):
            if (row_idx, col_idx) in skip_cells:
                # For skipped cells in markdown, we'll use empty string
                cell_value = ""
            else:
                cell_value = texts[row_idx][col_idx]

            row_cells.append(str(cell_value))  # Ensure string for join

        # Write the row
        markdown += "| " + " | ".join(row_cells) + " |\n"

    return markdown


def table_to_html(table_dict, font_family='Arial', header_bg_color='#f0f0f0', header_text_color='#000000'):
    texts = table_dict['texts']
    merged_regions = table_dict['merged_regions']
    top_header_rows = table_dict['top_header_rows_num']
    left_header_cols = table_dict['left_header_columns_num']
    title = table_dict.get('title', '')

    num_rows = len(texts)
    num_cols = len(texts[0]) if texts else 0

    merged_cells = {}
    skip_cells = set()

    for region in merged_regions:
        first_row, last_row = region['first_row'], region['last_row']
        first_col, last_col = region['first_column'], region['last_column']

        rowspan = last_row - first_row + 1
        colspan = last_col - first_col + 1

        merged_cells[(first_row, first_col)] = (rowspan, colspan)

        for r in range(first_row, last_row + 1):
            for c in range(first_col, last_col + 1):
                if r != first_row or c != first_col:
                    skip_cells.add((r, c))

    html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: {font_family}, sans-serif;
                padding: 20px;
                background-color: white;
            }}
            table {{
                border-collapse: collapse;
                margin: 0 auto;
                background-color: white;
                font-family: {font_family}, sans-serif;
            }}
            td, th {{
                border: 1px solid #333;
                padding: 8px 12px;
                text-align: center;
                font-size: 14px;
                font-family: {font_family}, sans-serif;
            }}
            .header-cell {{
                background-color: {header_bg_color};
                color: {header_text_color};
                font-weight: bold;
            }}
            .row-label, .col-label {{
                background-color: #e0e0e0;
                font-weight: bold;
                font-size: 12px;
                color: #333;
            }}
            .table-title {{
                font-size: 16px;
                font-weight: bold;
                text-align: center;
                font-family: {font_family}, sans-serif;
            }}
        </style>
    </head>
    <body>
        <table>
    """

    # Row counter starts at 1
    current_row = 1

    # Column labels row (row number + A, B, C, ...)
    html += "        <tr>\n"
    html += f'            <td class="row-label">{current_row}</td>\n'
    for col_idx in range(num_cols):
        html += f'            <td class="col-label">{column_letter(col_idx)}</td>\n'
    html += "        </tr>\n"
    current_row += 1

    # Title row if present (spans all columns)
    if title:
        html += "        <tr>\n"
        html += f'            <td class="row-label">{current_row}</td>\n'
        html += f'            <td class="table-title" colspan="{num_cols}">{title}</td>\n'
        html += "        </tr>\n"
        current_row += 1

    # Table content rows with row numbers continuing
    for row_idx in range(num_rows):
        html += "        <tr>\n"

        # Row number continues from current_row
        html += f'            <td class="row-label">{current_row}</td>\n'
        current_row += 1

        for col_idx in range(num_cols):
            if (row_idx, col_idx) in skip_cells:
                continue

            cell_value = texts[row_idx][col_idx]

            is_header = (row_idx < top_header_rows) or (col_idx < left_header_cols)
            cell_class = 'header-cell' if is_header else ''

            rowspan, colspan = merged_cells.get((row_idx, col_idx), (1, 1))

            span_attrs = ""
            if rowspan > 1:
                span_attrs += f' rowspan="{rowspan}"'
            if colspan > 1:
                span_attrs += f' colspan="{colspan}"'

            html += f'            <td class="{cell_class}"{span_attrs}>{cell_value}</td>\n'

        html += "        </tr>\n"

    html += """
        </table>
    </body>
    </html>
    """

    return html

VARIATIONS = {
    'arial': {
        'font_family': 'Arial',
        'header_bg_color': '#f0f0f0',
        'header_text_color': '#000000'
    },
    'times_new_roman': {
        'font_family': 'Times New Roman',
        'header_bg_color': '#f0f0f0',
        'header_text_color': '#000000'
    },
    'red': {
        'font_family': 'Arial',
        'header_bg_color': '#ff6b6b',
        'header_text_color': '#ffffff'
    },
    'blue': {
        'font_family': 'Arial',
        'header_bg_color': '#4a90e2',
        'header_text_color': '#ffffff'
    },
    'green': {
        'font_family': 'Arial',
        'header_bg_color': '#51cf66',
        'header_text_color': '#ffffff'
    }
}


async def html_to_image_base64(html_content):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.set_content(html_content)
        await page.wait_for_load_state('networkidle')

        screenshot_bytes = await page.screenshot(type='png', full_page=True)

        await browser.close()

        return base64.b64encode(screenshot_bytes).decode('utf-8')


def assign_splits(num_samples):
    indices = list(range(num_samples))
    random.shuffle(indices)

    train_end = int(num_samples * 0.80)
    val_end = train_end + int(num_samples * 0.066)
    dev_end = val_end + int(num_samples * 0.066)

    splits = [''] * num_samples
    for i in indices[:train_end]:
        splits[i] = 'train'
    for i in indices[train_end:val_end]:
        splits[i] = 'validation'
    for i in indices[val_end:dev_end]:
        splits[i] = 'dev'
    for i in indices[dev_end:]:
        splits[i] = 'test'

    return splits


async def generate_table_images(table_dict):
    images = {}

    for variation_name, style_params in VARIATIONS.items():
        html = table_to_html(
            table_dict,
            font_family=style_params['font_family'],
            header_bg_color=style_params['header_bg_color'],
            header_text_color=style_params['header_text_color']
        )
        images[variation_name] = await html_to_image_base64(html)

    return images


async def process_dataset_to_jsonl(dataset, output_path, test_mode=False, save_test_images=False):
    print(f"\n{'='*80}")
    print(f"Processing HiTab dataset to JSONL...")
    print(f"{'='*80}")

    # Collect all samples from all splits
    all_samples = []
    for split_name in ['train', 'validation', 'test']:
        for sample in dataset[split_name]:
            sample_data = {
                'question': sample['question'],
                'answer': sample['answer'],
                'answer_formulas': sample.get('answer_formulas', []),
                'table_id': sample['table_id'],
                'table_content': sample['table_content'],
                'source_split': f"hitab_{split_name}",  # Original HiTab split
                'highlighted_cells': sample.get('highlighted_cells', []),
            }
            all_samples.append(sample_data)

    print(f"Found {len(all_samples)} total samples")

    # Limit in test mode
    if test_mode:
        all_samples = all_samples[:20]
        print(f"TEST MODE: Processing only {len(all_samples)} samples")

    # Assign new splits
    splits = assign_splits(len(all_samples))

    # Cache for table data (markdown and images) by table_id
    table_cache = {}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup test image directories if needed
    test_img_dir = None
    test_md_dir = None
    if save_test_images:
        test_img_dir = output_path.parent / 'test_images'
        test_md_dir = output_path.parent / 'test_md'
        test_img_dir.mkdir(exist_ok=True)
        test_md_dir.mkdir(exist_ok=True)
        print(f"Saving test images to: {test_img_dir}/")
        print(f"Saving test markdown to: {test_md_dir}/")

    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, sample in enumerate(tqdm(all_samples, desc="Processing samples")):
            try:
                table_id = sample['table_id']

                # Generate or retrieve cached table data
                if table_id not in table_cache:
                    table_dict = parse_table_content(sample['table_content'])

                    # Generate markdown
                    markdown = table_to_markdown(table_dict)

                    # Generate all image variations
                    images = await generate_table_images(table_dict)

                    table_cache[table_id] = {
                        'markdown': markdown,
                        'images': images,
                        'json': table_dict
                    }

                    # Save test files if in test mode
                    if save_test_images:
                        # Save markdown
                        md_file = test_md_dir / f"{table_id}.md"
                        md_file.write_text(markdown, encoding='utf-8')

                        # Save images
                        for var_name, img_b64 in images.items():
                            var_dir = test_img_dir / var_name
                            var_dir.mkdir(exist_ok=True)
                            img_file = var_dir / f"{table_id}.png"
                            img_file.write_bytes(base64.b64decode(img_b64))

                # Build JSONL entry
                entry = {
                    'id': f"visualcite_{idx:06d}",
                    'split': splits[idx],
                    'question': sample['question'],
                    'answer': sample['answer'],
                    'answer_formulas': sample['answer_formulas'],
                    'highlighted_cells': sample['highlighted_cells'],
                    'table_json': table_cache[table_id]['json'],
                    'table_md': table_cache[table_id]['markdown'],
                    'table_images': table_cache[table_id]['images'],
                    'source': sample['source_split'],
                    'source_id': table_id
                }

                # Write to JSONL
                f.write(json.dumps(entry) + '\n')

            except Exception as e:
                print(f"\nError processing sample {idx} (table {sample.get('table_id')}): {e}")
                continue

    print(f"\n✓ JSONL file saved to: {output_path}")

    # Print split statistics
    split_counts = {'train': 0, 'validation': 0, 'dev': 0, 'test': 0}
    for split in splits[:len(all_samples)]:
        split_counts[split] += 1

    print("\nSplit distribution:")
    total = sum(split_counts.values())
    for split, count in split_counts.items():
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {split}: {count} ({pct:.1f}%)")

    # Create dataset_info.json
    dataset_info = {
        "dataset_name": "VisualCite",
        "description": "A table question answering dataset with visual perturbations, derived from HiTab",
        "version": "1.0.0",
        "citation": "Extended from HiTab (Cheng et al., 2022)",
        "homepage": "https://github.com/Yahialqur/VisualCite",
        "license": "Same as HiTab",
        "features": {
            "id": "Unique sample identifier",
            "split": "Dataset split (train/validation/dev/test)",
            "question": "Natural language question about the table",
            "answer": "Answer to the question",
            "answer_formulas": "Formulas used to derive the answer",
            "highlighted_cells": "Cells relevant to answering the question",
            "table_json": "Table structure in JSON format",
            "table_md": "Table in markdown format with row/column labels",
            "table_images": "Table images in 5 visual variations (base64 PNG)",
            "source": "Original HiTab split",
            "source_id": "Original table ID from HiTab"
        },
        "splits": split_counts,
        "total_samples": total,
        "num_unique_tables": len(table_cache),
        "image_variations": list(VARIATIONS.keys()),
        "random_seed": random.getstate()[1][0]  # Get the seed used
    }

    info_path = output_path.parent / 'dataset_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2)
    print(f"\n✓ Dataset info saved to: {info_path}")

    return split_counts


async def main():
    parser = argparse.ArgumentParser(description='Generate VisualCite dataset from HiTab')
    parser.add_argument('--test-mode', action='store_true',
                       help='Process only 20 samples for testing')
    parser.add_argument('--save-test-images', action='store_true',
                       help='Save test images and markdown files for verification')
    parser.add_argument('--output', type=str, default='../data/visualcite.jsonl',
                       help='Output JSONL file path')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for split assignment')
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    output_path = Path(args.output)

    print("="*80)
    print("VisualCite Dataset Generator")
    print("="*80)
    print(f"Output file: {output_path}")
    print(f"Test mode: {args.test_mode}")
    print(f"Save test images: {args.save_test_images}")
    print(f"Random seed: {args.seed}")

    print("\nLoading HiTab dataset...")
    dataset = load_dataset('kasnerz/hitab')
    print("✓ Dataset loaded")

    await process_dataset_to_jsonl(dataset, output_path,
                                   test_mode=args.test_mode,
                                   save_test_images=args.save_test_images)

    print("\n" + "="*80)
    print("ALL PROCESSING COMPLETE!")
    print("="*80)
    print(f"\nDataset saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")

    # Count lines
    with open(output_path, 'r') as f:
        line_count = sum(1 for _ in f)
    print(f"Total samples: {line_count}")


if __name__ == "__main__":
    asyncio.run(main())