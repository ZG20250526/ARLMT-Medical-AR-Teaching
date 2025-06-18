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


## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.8 or higher (for GPU acceleration)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/ZG20250526/ARLMT-Medical-AR-Teaching.git
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

### Recommended Requirements
- **CPU**: Intel i7-10900X or higher 
- **GPU**: NVIDIA RTX 3090 24GB or higher
- **RAM**: 128GB DDR4
- **Storage**: 1TB SSD

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
  title={Research on the application of LLaVA model based on Q-LoRA fine-tuning in medical teaching},
  author={Shiling Zhou and Fengmei Qin},
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

- Training scripts
- Evaluation benchmarks
- AR device access modules
- Documentation and examples

## Acknowledgments

- LLaVA team for the foundational multimodal architecture
- Medical imaging datasets providers
- INMO for AR hardware support
- Research collaborators and institutions

## Contact

- **Corresponding Author**: Shiling Zhou (3180100017@caa.edu.cn)
- **Institution**: School of Big Data and Information Industry, CCMC, Chongqing, China

## Funding

The General Project of the 2024 Special Education Reform Research of the Chongqing Education Science Planning Project(Grant No. K24ZG3330238)

---

**Note**: This repository contains the complete implementation of the ARLMT system as described in our PLOS ONE publication. For questions about commercial licensing or collaboration opportunities, please contact the corresponding author.
