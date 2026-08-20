#!/usr/bin/env python3
"""
Test utilities for validating the benchmark setup without running full inference.
"""

import sys
import argparse
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_data_loader(max_samples: int = 5):
    """Test that the dataset can be loaded"""
    logger.info("\n=== Testing Data Loader ===")
    
    from data_loader import VisualCiteDataset, parse_cell_coordinates
    
    dataset = VisualCiteDataset(
        jsonl_path="../visualcite.jsonl",
        split="dev",
        max_samples=max_samples
    )
    
    try:
        dataset.load()
        logger.info(f"✓ Dataset loaded successfully: {len(dataset)} samples")
        
        # Test first sample
        sample = dataset[0]
        logger.info(f"\n  Sample ID: {sample.id}")
        logger.info(f"  Question: {sample.question[:80]}...")
        logger.info(f"  Answer: {sample.answer}")
        logger.info(f"  Highlighted cells (indices): {sample.highlighted_cells}")
        logger.info(f"  Highlighted cells (coords): {parse_cell_coordinates(sample.highlighted_cells)}")
        logger.info(f"  Has markdown: {bool(sample.table_md)}")
        logger.info(f"  Has JSON: {bool(sample.table_json)}")
        logger.info(f"  Image styles: {list(sample.table_images.keys())}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Failed to load dataset: {e}")
        return False


def test_metrics():
    """Test metrics calculation"""
    logger.info("\n=== Testing Metrics ===")
    
    from metrics import evaluate_single_prediction, aggregate_metrics
    
    # Test case 1: Perfect match
    pred1 = ["A1", "B2"]
    gt1 = ["A1", "B2"]
    m1 = evaluate_single_prediction(pred1, gt1)
    assert m1.exact_match, "Expected exact match"
    assert m1.cell_f1 == 1.0, f"Expected F1=1.0, got {m1.cell_f1}"
    logger.info("✓ Perfect match test passed")
    
    # Test case 2: Partial match
    pred2 = ["A1", "C3"]
    gt2 = ["A1", "B2"]
    m2 = evaluate_single_prediction(pred2, gt2)
    assert m2.partial_match, "Expected partial match"
    assert not m2.exact_match, "Should not be exact match"
    assert m2.cell_precision == 0.5, f"Expected precision=0.5, got {m2.cell_precision}"
    logger.info("✓ Partial match test passed")
    
    # Test case 3: No match
    pred3 = ["X9"]
    gt3 = ["A1"]
    m3 = evaluate_single_prediction(pred3, gt3)
    assert not m3.partial_match, "Should not match"
    assert m3.cell_f1 == 0.0, f"Expected F1=0.0, got {m3.cell_f1}"
    logger.info("✓ No match test passed")
    
    # Test aggregation
    agg = aggregate_metrics([m1, m2, m3])
    assert agg.total_samples == 3
    logger.info(f"✓ Aggregation test passed (mean F1: {agg.mean_cell_f1:.3f})")
    
    return True


def test_prompt_builder():
    """Test prompt generation"""
    logger.info("\n=== Testing Prompt Builder ===")
    
    from prompt_builder import PromptBuilder
    from config import PromptStrategy, DataRepresentation
    from data_loader import VisualCiteSample
    
    # Create mock sample
    sample = VisualCiteSample(
        id="test_001",
        split="test",
        question="What was the total revenue?",
        answer="$100",
        answer_formulas=[],
        highlighted_cells=[[2, 1]],
        table_json={"texts": [["Header", "Value"], ["Revenue", "$100"]]},
        table_md="| | A | B |\n| 1 | Header | Value |\n| 2 | Revenue | $100 |",
        table_images={},
        source="test",
        source_id="test_001"
    )
    
    builder = PromptBuilder()
    
    # Test zero-shot
    prompt_zs, example_img_zs = builder.build_prompt(
        sample=sample,
        strategy=PromptStrategy.ZERO_SHOT,
        representation=DataRepresentation.MARKDOWN,
        table_content=sample.table_md
    )
    assert "ATTRIBUTED CELLS:" in prompt_zs
    assert sample.question in prompt_zs
    assert example_img_zs is None
    logger.info("✓ Zero-shot prompt generated")
    
    # Test few-shot
    prompt_fs, example_img_fs = builder.build_prompt(
        sample=sample,
        strategy=PromptStrategy.FEW_SHOT,
        representation=DataRepresentation.MARKDOWN,
        table_content=sample.table_md
    )
    assert "EXAMPLE:" in prompt_fs
    assert isinstance(prompt_fs, str)
    assert example_img_fs is None
    logger.info("✓ Few-shot prompt generated")
    
    # Test CoT
    prompt_cot, example_img_cot = builder.build_prompt(
        sample=sample,
        strategy=PromptStrategy.CHAIN_OF_THOUGHT,
        representation=DataRepresentation.MARKDOWN,
        table_content=sample.table_md
    )
    assert "step by step" in prompt_cot.lower()
    assert example_img_cot is None
    logger.info("✓ Chain-of-thought prompt generated")
    
    return True


def test_output_parsing():
    """Test model output parsing"""
    logger.info("\n=== Testing Output Parsing ===")
    
    from data_loader import parse_model_output
    
    test_cases = [
        ("A1, B2, C3", ["A1", "B2", "C3"]),
        ("ATTRIBUTED CELLS: A1", ["A1"]),
        ("The answer is in cell B3 and also C4.", ["B3", "C4"]),
        ("Based on the table, cells AA1 and AB2 contain the answer", ["AA1", "AB2"]),
        ("A1\nB2\nC3", ["A1", "B2", "C3"]),
        ("", []),
        ("No cells found", []),
    ]
    
    for input_text, expected in test_cases:
        result = parse_model_output(input_text)
        assert result == expected, f"For '{input_text}': expected {expected}, got {result}"
        logger.info(f"✓ Parse test passed: '{input_text[:30]}...' -> {result}")
    
    return True


def test_checkpoint_manager():
    """Test checkpoint save/load"""
    logger.info("\n=== Testing Checkpoint Manager ===")
    
    import tempfile
    import shutil
    from checkpoint_manager import CheckpointManager, BenchmarkProgress
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        manager = CheckpointManager(temp_dir)
        
        # Create and save progress
        progress = manager.create_progress(
            model_name="test_model",
            representation="json",
            strategy="zero_shot",
            total_samples=100
        )
        
        # Add some results
        progress.completed_sample_ids.append("sample_001")
        progress.results.append({
            "sample_id": "sample_001",
            "predicted_cells": ["A1"],
            "ground_truth_cells": ["A1"],
            "cell_f1": 1.0
        })
        
        manager.save_progress(progress)
        logger.info("✓ Checkpoint saved")
        
        # Clear cache and reload
        manager._progress_cache.clear()
        loaded = manager.load_progress("test_model", "json", "zero_shot")
        
        assert loaded is not None, "Failed to load checkpoint"
        assert len(loaded.completed_sample_ids) == 1
        assert loaded.completed_sample_ids[0] == "sample_001"
        logger.info("✓ Checkpoint loaded and verified")
        
        # Test clear
        manager.clear_checkpoint("test_model", "json", "zero_shot")
        loaded2 = manager.load_progress("test_model", "json", "zero_shot")
        assert loaded2 is None, "Checkpoint should be cleared"
        logger.info("✓ Checkpoint cleared successfully")
        
        return True
    
    finally:
        shutil.rmtree(temp_dir)


def test_model_availability():
    """Check if models can be loaded (dry run)"""
    logger.info("\n=== Testing Model Availability ===")
    
    try:
        from transformers import AutoConfig
        
        models = [
            "Qwen/Qwen3-VL-2B-Instruct",
            "Qwen/Qwen3-VL-4B-Instruct",
            "Qwen/Qwen3-VL-8B-Instruct",
            "Qwen/Qwen3-VL-32B-Instruct",
        ]
        
        for model_name in models:
            try:
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
                logger.info(f"✓ {model_name} - config accessible")
            except Exception as e:
                logger.warning(f"✗ {model_name} - {e}")
        
        return True
    except ImportError:
        logger.warning("transformers not installed, skipping model check")
        return True


def run_all_tests():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("VisualCite Benchmark Test Suite")
    logger.info("=" * 60)
    
    results = []
    
    # Run tests in order
    tests = [
        ("Metrics", test_metrics),
        ("Prompt Builder", test_prompt_builder),
        ("Output Parsing", test_output_parsing),
        ("Checkpoint Manager", test_checkpoint_manager),
        ("Data Loader", lambda: test_data_loader(3)),
        ("Model Availability", test_model_availability),
    ]
    
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            logger.error(f"\n✗ {name} failed with error: {e}")
            results.append((name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"  {status}: {name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


def main():
    parser = argparse.ArgumentParser(description="Test VisualCite benchmark components")
    parser.add_argument('--test', choices=['all', 'data', 'metrics', 'prompts', 'parsing', 'checkpoint', 'models'],
                       default='all', help='Which test to run')
    parser.add_argument('--max-samples', type=int, default=3, help='Max samples for data test')
    
    args = parser.parse_args()
    
    if args.test == 'all':
        success = run_all_tests()
    elif args.test == 'data':
        success = test_data_loader(args.max_samples)
    elif args.test == 'metrics':
        success = test_metrics()
    elif args.test == 'prompts':
        success = test_prompt_builder()
    elif args.test == 'parsing':
        success = test_output_parsing()
    elif args.test == 'checkpoint':
        success = test_checkpoint_manager()
    elif args.test == 'models':
        success = test_model_availability()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()