# ARLMT: Augmented Reality Large Language Model Medical Teaching System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![arXiv](https://img.shields.io/badge/arXiv-2025.XXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

## Overview

The Augmented Reality Large Language Model Medical Teaching System (ARLMT) integrates Augmented Reality (AR) with a fine-tuned Large Language and Vision Assistant for Medicine (LLaVA-Med), employing Quantized Low-Rank Adaptation (QLoRA) to advance medical education. This system is designed for deployment on resource-constrained AR devices such as INMO Air2 glasses.

## Key Features

- **Memory Efficiency**: 66% reduction in memory footprint (from 15.2 GB to 5.1 GB) through QLoRA
- **High Accuracy**: 98.3% diagnostic accuracy in medical imaging tasks
- **Real-time Performance**: Average response time of 1.009 seconds
- **AR Integration**: Seamless deployment on INMO Air2 AR glasses
- **Medical Specialization**: Fine-tuned on medical datasets for biomedical applications

## System Architecture

```
ARLMT System
├── AR Interface Layer (INMO Air2 Glasses)
├── Computer Vision Module (Real-time Scene Capture)
├── LLaVA-Med Model (QLoRA Fine-tuned)
├── Medical Knowledge Base
└── Feedback Generation System
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.8 or higher (for GPU acceleration)
- 8GB+ RAM (16GB+ recommended)
- INMO Air2 AR glasses (for full AR functionality)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/YourUsername/ARLMT.git
cd ARLMT

# Create virtual environment
conda create -n arlmt python=3.8
conda activate arlmt

# Install dependencies
pip install -r requirements.txt

# Install additional packages for AR integration
pip install opencv-python pyopengl pygame
```

### Model Setup

```bash
# Download pre-trained LLaVA-Med model
python scripts/download_models.py

# Apply QLoRA fine-tuning (optional - pre-trained weights available)
python scripts/apply_qlora.py --config configs/qlora_config.yaml
```

## Quick Start

### Basic Usage

```python
from arlmt import ARLMTSystem

# Initialize the system
arlmt = ARLMTSystem(
    model_path="models/llava_med_qlora",
    ar_device="inmo_air2"
)

# Start AR medical teaching session
arlmt.start_session()

# Process medical image with AR overlay
result = arlmt.process_medical_image(
    image_path="data/sample_xray.jpg",
    question="What abnormalities do you observe?"
)

print(f"Diagnosis: {result['diagnosis']}")
print(f"Confidence: {result['confidence']}")
print(f"Response Time: {result['response_time']}s")
```

### AR Glasses Integration

```python
# Connect to INMO Air2 glasses
arlmt.connect_ar_device()

# Enable real-time medical scene analysis
arlmt.enable_realtime_analysis(
    overlay_annotations=True,
    voice_feedback=True,
    haptic_feedback=False
)
```

## Dataset

The ARLMT system is trained on a combination of medical datasets:

- **LLaVA-Med Dataset**: Medical visual question answering pairs
- **Medical Imaging Collections**: X-rays, CT scans, MRI images
- **Pathology Images**: Histopathology and cytopathology samples
- **Custom AR Scenarios**: Simulated medical teaching environments

### Data Access

Due to privacy regulations and ethical considerations, medical datasets are available through controlled access:

1. **Research Access**: Contact the corresponding author for research collaboration
2. **Educational Use**: Limited datasets available for educational institutions
3. **Commercial Licensing**: Contact technology transfer office for commercial applications

## Model Performance

| Metric | ARLMT (QLoRA) | GPT-4 Baseline | Improvement |
|--------|---------------|----------------|-------------|
| Diagnostic Accuracy | 98.3% | 93.1% | +5.2% |
| Response Time | 1.009s | 1.176s | +14.2% |
| Memory Usage | 5.1GB | 15.2GB | -66.4% |
| Energy Efficiency | 2.3W | 8.7W | -73.6% |

## Evaluation

### Running Evaluations

```bash
# Evaluate on medical VQA tasks
python eval/evaluate_medical_vqa.py --dataset slake --model_path models/llava_med_qlora

# Benchmark AR performance
python eval/benchmark_ar_performance.py --device inmo_air2

# Measure system latency
python eval/measure_latency.py --iterations 1000
```

### Evaluation Metrics

- **Medical Accuracy**: Diagnostic precision on medical imaging tasks
- **Response Latency**: Time from image capture to feedback generation
- **Memory Efficiency**: Peak memory usage during inference
- **User Experience**: Measured via NASA-TLX and Presence Questionnaire

## Hardware Requirements

### Minimum Requirements
- **CPU**: Intel i5-8400 or AMD Ryzen 5 2600
- **GPU**: NVIDIA GTX 1060 6GB or equivalent
- **RAM**: 8GB DDR4
- **Storage**: 50GB available space

### Recommended Requirements
- **CPU**: Intel i7-10700K or AMD Ryzen 7 3700X
- **GPU**: NVIDIA RTX 3070 8GB or higher
- **RAM**: 16GB DDR4
- **Storage**: 100GB SSD

### AR Device Compatibility
- **Primary**: INMO Air2 AR Glasses
- **Secondary**: Microsoft HoloLens 2
- **Experimental**: Magic Leap 2

## Contributing

We welcome contributions from the research community. Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black arlmt/
flake8 arlmt/
```

## Citation

If you use ARLMT in your research, please cite our paper:

```bibtex
@article{arlmt2025,
  title={Application of Augmented Reality System Based on LLaVA in Medical Teaching and QLoRA Fine-tuning},
  author={Cat and Dog},
  journal={PLOS ONE},
  year={2025},
  volume={XX},
  number={X},
  pages={eXXXXXXX},
  doi={10.1371/journal.pone.XXXXXXX}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Data Availability Statement

The datasets supporting the conclusions of this article are available through controlled access due to privacy and ethical considerations. Researchers interested in accessing the data should contact the corresponding author at 3180100017@caa.edu.cn with a detailed research proposal and institutional approval.

## Code Availability

All source code is available in this repository under the MIT license. The code is designed to be reproducible and includes:

- Complete training scripts
- Evaluation benchmarks
- AR integration modules
- Documentation and examples

## Acknowledgments

- LLaVA team for the foundational multimodal architecture
- Medical imaging datasets providers
- INMO for AR hardware support
- Research collaborators and institutions

## Contact

- **Corresponding Author**: Dog (3180100017@caa.edu.cn)
- **Institution**: School of Big Data and Information Industry, CCMC, Chongqing, China
- **Project Website**: [https://arlmt-project.github.io](https://arlmt-project.github.io)

## Funding

This research was supported by [Grant Information - to be added based on actual funding sources].

---

**Note**: This repository contains the complete implementation of the ARLMT system as described in our PLOS ONE publication. For questions about commercial licensing or collaboration opportunities, please contact the corresponding author.