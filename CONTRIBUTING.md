# Contributing to ARLMT

We welcome contributions to the Augmented Reality Large Language Model Medical Teaching System (ARLMT) project! This document provides guidelines for contributing to ensure a collaborative and productive environment.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Types of Contributions](#types-of-contributions)
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Medical Data and Ethics](#medical-data-and-ethics)
- [Review Process](#review-process)
- [Recognition](#recognition)

## Code of Conduct

### Our Commitment

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background, experience level, or affiliation. We expect all participants to:

- Use welcoming and inclusive language
- Respect differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the community and medical education
- Show empathy towards other community members

### Medical Ethics

Given the medical nature of this project, we additionally require:

- Respect for patient privacy and confidentiality
- Adherence to medical research ethics
- Responsible AI development practices
- Clear disclaimers about educational vs. clinical use

### Enforcement

Instances of unacceptable behavior may be reported to the project maintainers at 3180100017@caa.edu.cn. All complaints will be reviewed and investigated promptly and fairly.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git version control
- Basic understanding of machine learning and computer vision
- Familiarity with PyTorch and Transformers library
- (Optional) Experience with AR development

### First Steps

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YourUsername/ARLMT.git
   cd ARLMT
   ```

2. **Set Up Development Environment**
   ```bash
   # Create virtual environment
   conda create -n arlmt-dev python=3.8
   conda activate arlmt-dev
   
   # Install dependencies
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Verify Installation**
   ```bash
   # Run basic tests
   python -m pytest tests/test_basic.py
   
   # Check code style
   black --check arlmt/
   flake8 arlmt/
   ```

## Types of Contributions

### 🐛 Bug Reports

**Before submitting a bug report:**
- Check existing issues to avoid duplicates
- Test with the latest version
- Gather relevant system information

**Bug report should include:**
- Clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- System information (OS, Python version, GPU, etc.)
- Error messages and stack traces
- Minimal code example (if applicable)

### 💡 Feature Requests

**Good feature requests:**
- Address a clear need in medical education
- Are technically feasible
- Align with project goals
- Include implementation suggestions

**Feature request template:**
- Problem description
- Proposed solution
- Alternative solutions considered
- Additional context or examples

### 🔧 Code Contributions

**Areas where contributions are especially welcome:**
- AR interface improvements
- Model optimization techniques
- Medical dataset integration
- Performance benchmarking
- Documentation and examples
- Testing and quality assurance

### 📚 Documentation

**Documentation contributions:**
- API documentation
- Tutorials and examples
- Installation guides
- Best practices
- Translation to other languages

### 🧪 Research Contributions

**Research-oriented contributions:**
- Novel AR interaction paradigms
- Medical education effectiveness studies
- Model architecture improvements
- Evaluation metrics and benchmarks

## Development Setup

### Environment Configuration

```bash
# Clone and setup
git clone https://github.com/YourUsername/ARLMT.git
cd ARLMT

# Create development environment
conda create -n arlmt-dev python=3.8
conda activate arlmt-dev

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install
```

### Development Tools

**Code Quality:**
- `black`: Code formatting
- `flake8`: Linting
- `isort`: Import sorting
- `mypy`: Type checking

**Testing:**
- `pytest`: Test framework
- `pytest-cov`: Coverage reporting
- `pytest-mock`: Mocking utilities

**Documentation:**
- `sphinx`: Documentation generation
- `sphinx-rtd-theme`: Documentation theme

### IDE Configuration

**VS Code (Recommended):**
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true
}
```

## Contribution Workflow

### 1. Issue Creation

```bash
# Create issue on GitHub first
# Reference issue number in commits and PRs
```

### 2. Branch Creation

```bash
# Create feature branch
git checkout -b feature/issue-123-ar-optimization

# Or bug fix branch
git checkout -b bugfix/issue-456-memory-leak
```

### 3. Development

```bash
# Make changes
# Write tests
# Update documentation

# Run tests frequently
python -m pytest tests/

# Check code quality
black arlmt/
flake8 arlmt/
isort arlmt/
```

### 4. Commit Guidelines

**Commit Message Format:**
```
type(scope): brief description

Detailed explanation of changes (if needed)

Fixes #123
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions or modifications
- `chore`: Maintenance tasks

**Examples:**
```bash
git commit -m "feat(ar): add INMO Air2 gesture recognition

Implemented hand gesture detection for AR interface
using MediaPipe. Supports basic navigation gestures.

Fixes #123"

git commit -m "fix(model): resolve QLoRA memory leak

Fixed memory accumulation in QLoRA fine-tuning loop.
Reduced memory usage by 15% during training.

Fixes #456"
```

### 5. Pull Request

**Before submitting:**
- Ensure all tests pass
- Update documentation
- Add changelog entry
- Rebase on latest main branch

**PR Template:**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Medical Ethics Compliance
- [ ] No patient data exposed
- [ ] Educational use clearly documented
- [ ] Privacy considerations addressed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## Coding Standards

### Python Style

**Follow PEP 8 with these specifics:**
- Line length: 88 characters (Black default)
- Use type hints for all functions
- Docstrings for all public functions and classes
- Meaningful variable and function names

**Example:**
```python
from typing import List, Optional, Tuple
import torch
from transformers import AutoModel


class ARLMTModel:
    """ARLMT model for medical AR applications.
    
    This class implements the core ARLMT functionality,
    combining LLaVA-Med with QLoRA optimization for
    deployment on AR devices.
    
    Args:
        model_path: Path to pre-trained model
        device: Computing device ('cuda' or 'cpu')
        quantization: Whether to use QLoRA quantization
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        quantization: bool = True
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.quantization = quantization
        self._model: Optional[AutoModel] = None
    
    def process_medical_image(
        self,
        image: torch.Tensor,
        question: str
    ) -> Tuple[str, float]:
        """Process medical image with question.
        
        Args:
            image: Input medical image tensor
            question: Medical question about the image
            
        Returns:
            Tuple of (answer, confidence_score)
            
        Raises:
            ValueError: If image format is invalid
            RuntimeError: If model inference fails
        """
        if image.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {image.dim()}D")
        
        # Implementation here
        answer = "Sample diagnosis"
        confidence = 0.95
        
        return answer, confidence
```

### Documentation Standards

**Docstring Format (Google Style):**
```python
def train_model(
    model: ARLMTModel,
    dataset: Dataset,
    epochs: int = 10,
    learning_rate: float = 1e-4
) -> Dict[str, float]:
    """Train ARLMT model on medical dataset.
    
    This function implements the complete training pipeline
    for ARLMT, including QLoRA fine-tuning and evaluation.
    
    Args:
        model: Pre-initialized ARLMT model
        dataset: Training dataset with medical images and questions
        epochs: Number of training epochs (default: 10)
        learning_rate: Learning rate for optimization (default: 1e-4)
        
    Returns:
        Dictionary containing training metrics:
        - 'loss': Final training loss
        - 'accuracy': Final accuracy on validation set
        - 'training_time': Total training time in seconds
        
    Raises:
        ValueError: If dataset is empty or invalid
        RuntimeError: If training fails due to hardware issues
        
    Example:
        >>> model = ARLMTModel("path/to/model")
        >>> dataset = load_medical_dataset("data/")
        >>> metrics = train_model(model, dataset, epochs=5)
        >>> print(f"Final accuracy: {metrics['accuracy']:.2f}")
        Final accuracy: 0.98
        
    Note:
        This function requires significant GPU memory (>8GB)
        for optimal performance. Consider using gradient
        checkpointing for memory-constrained environments.
    """
    # Implementation here
    pass
```

### Testing Standards

**Test Structure:**
```python
import pytest
import torch
from unittest.mock import Mock, patch

from arlmt.model import ARLMTModel
from arlmt.utils import load_medical_image


class TestARLMTModel:
    """Test suite for ARLMTModel class."""
    
    @pytest.fixture
    def model(self):
        """Create test model instance."""
        return ARLMTModel(
            model_path="tests/fixtures/test_model",
            device="cpu",
            quantization=False
        )
    
    @pytest.fixture
    def sample_image(self):
        """Create sample medical image tensor."""
        return torch.randn(1, 3, 224, 224)
    
    def test_model_initialization(self, model):
        """Test model initializes correctly."""
        assert model.device == "cpu"
        assert model.quantization is False
        assert model._model is None
    
    def test_process_medical_image_valid_input(self, model, sample_image):
        """Test processing with valid inputs."""
        question = "What abnormalities do you see?"
        
        with patch.object(model, '_model') as mock_model:
            mock_model.generate.return_value = "No abnormalities detected"
            
            answer, confidence = model.process_medical_image(
                sample_image, question
            )
            
            assert isinstance(answer, str)
            assert 0.0 <= confidence <= 1.0
    
    def test_process_medical_image_invalid_dimensions(self, model):
        """Test error handling for invalid image dimensions."""
        invalid_image = torch.randn(3, 224, 224)  # Missing batch dimension
        question = "Test question"
        
        with pytest.raises(ValueError, match="Expected 4D tensor"):
            model.process_medical_image(invalid_image, question)
    
    @pytest.mark.slow
    def test_model_inference_performance(self, model, sample_image):
        """Test model inference performance."""
        import time
        
        question = "Analyze this medical image"
        
        start_time = time.time()
        answer, confidence = model.process_medical_image(
            sample_image, question
        )
        inference_time = time.time() - start_time
        
        # Should complete within 2 seconds on CPU
        assert inference_time < 2.0
        assert len(answer) > 0
```

## Testing Guidelines

### Test Categories

**Unit Tests:**
- Test individual functions and classes
- Fast execution (< 1 second each)
- No external dependencies
- High code coverage (>90%)

**Integration Tests:**
- Test component interactions
- May use real models/data
- Moderate execution time (< 30 seconds)
- Focus on critical paths

**End-to-End Tests:**
- Test complete workflows
- Use realistic scenarios
- Longer execution time (< 5 minutes)
- Validate user-facing functionality

**Performance Tests:**
- Benchmark critical operations
- Memory usage validation
- Latency measurements
- Marked with `@pytest.mark.slow`

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_model.py

# Run with coverage
python -m pytest --cov=arlmt --cov-report=html

# Run only fast tests
python -m pytest -m "not slow"

# Run with verbose output
python -m pytest -v
```

### Test Data

**Synthetic Data:**
- Use synthetic medical images for testing
- Generate realistic but artificial scenarios
- No privacy concerns

**Anonymized Data:**
- Limited use of anonymized medical data
- Proper ethical approval required
- Clear documentation of data source

**Mock Data:**
- Prefer mocking for external dependencies
- Faster test execution
- More reliable CI/CD

## Documentation

### Types of Documentation

**API Documentation:**
- Auto-generated from docstrings
- Complete parameter descriptions
- Usage examples
- Error conditions

**User Guides:**
- Installation instructions
- Quick start tutorials
- Common use cases
- Troubleshooting

**Developer Documentation:**
- Architecture overview
- Contributing guidelines
- Testing procedures
- Release process

**Research Documentation:**
- Model architecture details
- Training procedures
- Evaluation metrics
- Experimental results

### Documentation Build

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Build documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

## Medical Data and Ethics

### Data Handling Requirements

**Privacy Protection:**
- No personally identifiable information (PII)
- Proper anonymization procedures
- Secure data storage and transmission
- Access logging and monitoring

**Ethical Compliance:**
- IRB approval for research use
- Informed consent for data collection
- Clear data use agreements
- Regular ethics review

**Technical Safeguards:**
- Encryption at rest and in transit
- Access controls and authentication
- Audit trails for data access
- Secure deletion procedures

### Code Review for Medical Applications

**Additional Review Criteria:**
- Patient safety considerations
- Clinical accuracy validation
- Bias detection and mitigation
- Interpretability and explainability

**Medical Disclaimers:**
```python
# Required disclaimer for medical functions
def diagnose_medical_condition(image, symptoms):
    """
    MEDICAL DISCLAIMER:
    This function is for educational purposes only.
    It is not intended for clinical diagnosis or
    medical decision-making. Always consult with
    qualified healthcare professionals for medical
    advice and treatment.
    """
    # Implementation here
    pass
```

## Review Process

### Review Criteria

**Code Quality:**
- Follows coding standards
- Adequate test coverage
- Clear documentation
- Performance considerations

**Medical Appropriateness:**
- Clinically relevant
- Educationally sound
- Ethically compliant
- Safety considerations

**Technical Excellence:**
- Efficient implementation
- Proper error handling
- Security best practices
- Maintainable design

### Review Process

1. **Automated Checks**
   - Code style validation
   - Test execution
   - Security scanning
   - Documentation build

2. **Peer Review**
   - Technical review by maintainers
   - Medical review (if applicable)
   - User experience evaluation
   - Performance assessment

3. **Final Approval**
   - Maintainer approval required
   - Medical expert approval (for clinical features)
   - Security review (for sensitive changes)

### Review Timeline

- **Small fixes**: 1-3 days
- **Feature additions**: 1-2 weeks
- **Major changes**: 2-4 weeks
- **Medical features**: Additional time for expert review

## Recognition

### Contributor Recognition

**Contributors File:**
All contributors are acknowledged in `CONTRIBUTORS.md`

**Academic Recognition:**
- Significant contributors may be invited as co-authors
- Research contributions acknowledged in publications
- Conference presentation opportunities

**Community Recognition:**
- Contributor highlights in project updates
- Social media recognition
- Conference speaking opportunities

### Contribution Levels

**Code Contributors:**
- Bug fixes and improvements
- Feature development
- Performance optimization
- Testing and quality assurance

**Research Contributors:**
- Novel algorithms or techniques
- Evaluation studies
- Dataset contributions
- Theoretical insights

**Community Contributors:**
- Documentation improvements
- User support
- Educational content
- Outreach and advocacy

## Getting Help

### Support Channels

**GitHub Issues:**
- Bug reports and feature requests
- Technical discussions
- Public Q&A

**Email Support:**
- 3180100017@caa.edu.cn
- Private or sensitive inquiries
- Collaboration proposals

**Community Forums:**
- General discussions
- Best practices sharing
- User experiences

### Response Times

- **Critical bugs**: 24-48 hours
- **General issues**: 3-7 days
- **Feature requests**: 1-2 weeks
- **Research inquiries**: 1-2 weeks

---

**Thank you for contributing to ARLMT!** Your efforts help advance medical education through innovative AR and AI technologies. Together, we can create tools that improve learning outcomes and ultimately benefit patient care.

For questions about this contributing guide, please contact the maintainers at 3180100017@caa.edu.cn.