#!/usr/bin/env python3
"""
Model Evaluation Script for ARLMT System

This script evaluates the performance of trained ARLMT models
on various medical education benchmarks.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from evaluation.medical_qa_evaluator import MedicalQAEvaluator
from evaluation.ar_performance_evaluator import ARPerformanceEvaluator
from utils.logging_utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARLMT model")
    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--eval-data", 
        type=str, 
        required=True,
        help="Path to evaluation dataset"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./evaluation_results",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--eval-type", 
        type=str, 
        choices=["medical_qa", "ar_performance", "all"],
        default="all",
        help="Type of evaluation to perform"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.output_dir)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting ARLMT model evaluation...")
    
    results = {}
    
    # Medical QA evaluation
    if args.eval_type in ["medical_qa", "all"]:
        logger.info("Running Medical QA evaluation...")
        qa_evaluator = MedicalQAEvaluator(args.model_path)
        qa_results = qa_evaluator.evaluate(args.eval_data)
        results["medical_qa"] = qa_results
        logger.info(f"Medical QA Accuracy: {qa_results['accuracy']:.3f}")
    
    # AR Performance evaluation
    if args.eval_type in ["ar_performance", "all"]:
        logger.info("Running AR Performance evaluation...")
        ar_evaluator = ARPerformanceEvaluator(args.model_path)
        ar_results = ar_evaluator.evaluate(args.eval_data)
        results["ar_performance"] = ar_results
        logger.info(f"AR Response Time: {ar_results['avg_response_time']:.3f}ms")
    
    # Save results
    results_file = Path(args.output_dir) / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation completed! Results saved to {results_file}")


if __name__ == "__main__":
    main()
