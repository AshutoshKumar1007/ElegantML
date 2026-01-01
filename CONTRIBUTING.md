# Contributing to ElegantML

Thank you for your interest in contributing to ElegantML! This guide will help you get started.

## Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/AshutoshKumar1007/ElegantML.git
   cd ElegantML
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Unix/MacOS
   source venv/bin/activate
   ```

3. **Install in Development Mode**
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style Guidelines

### General Principles
- **Clarity over cleverness**: Code should be easy to understand.
- **NumPy-first**: All implementations should use NumPy, not external ML libraries
- **Mathematical transparency**: Stay close to the mathematical formulation
- **Comprehensive documentation**: All public functions need docstrings

### Python Style
- Follow PEP 8 style guide
- Use meaningful variable names
- Maximum line length: 100 characters
- Use type hints where appropriate

### Example Good Code
```python
def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid activation function.
    
    Parameters
    ----------
    x : ndarray
        Input array
        
    Returns
    -------
    ndarray
        Sigmoid of x, element-wise
        
    Notes
    -----
    sigmoid(x) = 1 / (1 + exp(-x))
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
```

## Adding New Features

### Adding a New Algorithm

1. **Choose the right module**
   - Neural network components → `elegantml/nn/`
   - Classical ML algorithms → `elegantml/ml/`
   - Clustering → `elegantml/clustering/`
   - Preprocessing → `elegantml/preprocessing/`
   - Metrics → `elegantml/metrics/`

2. **Implement the algorithm**
   ```python
   class MyNewAlgorithm:
       """
       Brief description of the algorithm.
       
       Parameters
       ----------
       param1 : type
           Description
       param2 : type, default=value
           Description
       """
       
       def __init__(self, param1, param2=default_value):
           self.param1 = param1
           self.param2 = param2
           
       def fit(self, X, y):
           """Fit the model to training data."""
           # Implementation
           return self
           
       def predict(self, X):
           """Make predictions on new data."""
           # Implementation
           return predictions
   ```

3. **Add to module __init__.py**
   ```python
   from elegantml.module_name import MyNewAlgorithm
   
   __all__ = [
       'ExistingClass',
       'MyNewAlgorithm',  # Add here
   ]
   ```

4. **Write tests** (see Testing section)

5. **Add example usage** in `examples/`

6. **Update README.md** with the new feature

### Testing

Every new feature must include tests:

1. **Create test file** (or add to existing)
   ```python
   # tests/test_my_feature.py
   import pytest
   import numpy as np
   from elegantml.module_name import MyNewAlgorithm
   
   def test_my_new_algorithm():
       """Test basic functionality."""
       X = np.random.randn(100, 5)
       y = np.random.randint(0, 2, 100)
       
       model = MyNewAlgorithm()
       model.fit(X, y)
       predictions = model.predict(X)
       
       assert predictions.shape == y.shape
       assert predictions.dtype == y.dtype
   ```

2. **Run tests**
   ```bash
   # Run all tests (if test suite exists)
   pytest tests/
   
   # Run specific test file
   pytest tests/test_my_feature.py -v
   ```

### Documentation

1. **Docstring Format** (NumPy style)
   ```python
   def function_name(param1, param2):
       """
       One-line summary.
       
       More detailed description if needed. Can span multiple
       lines and include mathematical formulas.
       
       Parameters
       ----------
       param1 : type
           Description of param1
       param2 : type, optional
           Description of param2
           
       Returns
       -------
       return_type
           Description of return value
           
       Examples
       --------
       >>> result = function_name(1, 2)
       >>> print(result)
       3
       
       Notes
       -----
       Any additional notes, mathematical formulas, or references.
       """
       pass
   ```

2. **Update README.md** when adding new features

3. **Add examples** showing how to use the new feature

## Pull Request Process

1. **Create a branch**
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes**
   - Write code
   - Add tests
   - Update documentation

3. **Test your changes**
   ```bash
   # Run tests (if available)
   pytest tests/
   
   # Test your code manually
   python -c "from elegantml.module_name import MyNewAlgorithm; print('Import works!')"
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add MyNewAlgorithm to module_name"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/my-new-feature
   ```

6. **Create Pull Request**
   - Go to GitHub
   - Click "New Pull Request"
   - Describe your changes
   - Reference any related issues

### PR Checklist
- [ ] Code follows PEP 8 style guidelines
- [ ] Docstrings are complete (NumPy style)
- [ ] Examples or notebooks are provided
- [ ] README.md is updated if needed
- [ ] No breaking changes (or clearly documented)
- [ ] Code has been tested manually

## Areas for Contribution

### High Priority
- [] Implement missing algorithms (e.g., Decision Trees, KNN)
- [ ] Comprehensive test suite
- [ ] Inbuilt datasets for testing and examples
- [ ] More unit tests for existing algorithms
- [ ] Random Forest implementation
- [ ] Relevance Vector Machines (RVM)


### Medium Priority
- [ ] Additional activation functions (GELU, Swish)
- [ ] Convolutional neural network layers
- [ ] Performance optimizations for neural network layers
- [ ] More clustering algorithms (DBSCAN, Hierarchical)
- [ ] Principal Component Analysis (PCA)
- [ ] Cross-validation utilities
- [ ] Data augmentation tools
- [ ] Model serialization improvements
- [ ] Visualization utilities

### Documentation
- [ ] Tutorial notebooks
- [ ] API reference documentation
- [ ] Mathematical explanations
- [ ] Performance benchmarks
- [ ] Comparison with scikit-learn

## Code Review Process

1. **Automated checks** run on all PRs
2. **Maintainer review** for code quality and design
3. **Discussion** if changes are needed
4. **Merge** once approved

## Questions?

- Open an issue for questions
- Tag with "question" label
- Be specific and provide context

## Code of Conduct

Be respectful, constructive, and collaborative. We're all here to learn and build something great together!

---

Thank you for contributing to ElegantML! 🎉
