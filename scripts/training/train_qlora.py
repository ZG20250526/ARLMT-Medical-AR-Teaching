#!/usr/bin/env python3
"""
QLoRA Training Script for ARLMT System

This script handles the fine-tuning of LLaVA-Med using QLoRA
for the ARLMT medical teaching system.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from arlmt_core.config import load_config
from qlora_implementation.trainer import QLoRATrainer
from utils.logging_utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train ARLMT with QLoRA")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./outputs",
        help="Output directory for models and logs"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.output_dir)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting ARLMT QLoRA training...")
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize trainer
    trainer = QLoRATrainer(config, args.output_dir)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed training from {args.resume}")
    
    # Start training
    trainer.train()
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
