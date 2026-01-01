# ElegantML

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ElegantML** is a NumPy-first machine learning library focused on clear, minimal, and mathematically grounded implementations of classical ML algorithms and neural networks.

## 🎯 Philosophy

ElegantML is designed for:
- **Education**: Clear, readable implementations that help understand ML algorithms from first principles
- **Transparency**: Every component is implemented from scratch using NumPy
- **Simplicity**: Minimal dependencies, maximum clarity
- **Mathematical Rigor**: Implementations stay true to the underlying mathematics

## 📦 Installation

### From PyPI (once published)
```bash
pip install elegantML
```

### From Source
```bash
git clone https://github.com/AshutoshKumar1007/elegantML.git
cd elegantML
pip install -e .
```

### Development Installation
```bash
pip install -e ".[dev]"
```

Then in Python:
```python
import elegantml
```
## 🚀 Quick Start

### Neural Networks

```python
import numpy as np
from elegantml.nn import Sequential, Linear, Relu, Softmax
from elegantml.nn import CrossEntropyLoss, Adam

# Create a simple neural network
model = Sequential([
    Linear(784, 128),
    Relu(),
    Linear(128, 64),
    Relu(),
    Linear(64, 10),
    Softmax()
])

# Define loss and optimizer
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    
    # Backward pass
    dloss = criterion.backward()
    model.backprop(dloss)
    
    # Update weights
    optimizer.step()
```

### Classical ML - Linear Regression

```python
from elegantml.ml import LinearRegression
import numpy as np

# Create and train model
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
```
<!-- 
### Classical ML - Decision Trees

```python
from elegantml.tree import DecisionTreeClassifier
from elegantml.metrics import accuracy_score

# Create and train model
model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
``` -->

### Clustering

```python
from elegantml.clustering import KMeans
import matplotlib.pyplot as plt

# Create and fit model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Get cluster assignments
labels = kmeans.labels_
centroids = kmeans.centroids

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200)
plt.show()
```

### Preprocessing

```python
from elegantml.preprocessing import StandardScaler, LabelEncoder

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
```

## 📚 Features

### Neural Network Components (`elegantml.nn`)
- **Layers**: Linear, BatchNorm, Dropout, Sequential
- **Activations**: ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
- **Optimizers**: BGD, BGD with Momentum, Adam
- **Losses**: MSE, Binary Cross-Entropy, Cross-Entropy, L2 Regularization
- **Features**: Model saving/loading, training/eval modes

### Classical ML Algorithms

#### Supervised ML (`elegantml.ml`)
- Linear Regression
- Logistic Regression
- Softmax Regression (multiclass)
- Support Vector Machines (Classifier, Regressor)
- Gaussian Process Regressor
- Relevance Vector Machine
- Gaussian Discriminant Analysis

#### Clustering (`elegantml.clustering`)
- K-Means Clustering

#### Preprocessing (`elegantml.preprocessing`)
- StandardScaler
- MinMaxScaler
- LabelEncoder
- OneHotEncoder

#### Metrics (`elegantml.metrics`)
- **Classification**: accuracy, precision, recall, F1-score, confusion matrix
- **Regression**: MSE, MAE, R² score

## 📖 Documentation

### Project Structure
```
ElegantML/
├── elegantml/
│   ├── __init__.py
│   ├── clustering/
│   │   └── __init__.py
│   ├── metrics/
│   │   └── __init__.py
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── gaussianProcess.py
│   │   ├── gda.py
│   │   ├── linearRegression.py
│   │   ├── logisticRegression.py
│   │   ├── rvm.py
│   │   ├── softmaxRegression.py
│   │   └── svm.py
│   ├── nn/
│   │   ├── __init__.py
│   │   ├── activations.py
│   │   ├── layers.py
│   │   ├── losses.py
│   │   └── optimizers.py
│   └── preprocessing/
│       └── __init__.py
├── notebooks/
│   ├── dev-history/
│   │   ├── GDA.ipynb
│   │   ├── linear_reg.ipynb
│   │   ├── logistic.ipynb
│   │   ├── neural.ipynb
│   │   ├── shocastic.ipynb
│   │   ├── softmax.ipynb
│   │   ├── SVm1.ipynb
│   │   ├── SVM2.ipynb
│   │   └── Gaussian Process/
│   │       ├── Gaussian.ipynb
│   │       ├── Gaussian2.ipynb
│   │       └── GP.ipynb
│   └── tutorials/
├── README.md
└── LICENSE
```

## 🔬 Examples

See the notebooks for worked examples and development history:
- [notebooks/dev-history/linear_reg.ipynb](notebooks/dev-history/linear_reg.ipynb)
- [notebooks/dev-history/logistic.ipynb](notebooks/dev-history/logistic.ipynb)
- [notebooks/dev-history/softmax.ipynb](notebooks/dev-history/softmax.ipynb)
- [notebooks/dev-history/GDA.ipynb](notebooks/dev-history/GDA.ipynb)
- [notebooks/dev-history/SVm1.ipynb](notebooks/dev-history/SVm1.ipynb)
- [notebooks/dev-history/SVM2.ipynb](notebooks/dev-history/SVM2.ipynb)
- [notebooks/dev-history/Gaussian%20Process/Gaussian.ipynb](notebooks/dev-history/Gaussian%20Process/Gaussian.ipynb)
- [notebooks/dev-history/Gaussian%20Process/Gaussian2.ipynb](notebooks/dev-history/Gaussian%20Process/Gaussian2.ipynb)
- [notebooks/dev-history/Gaussian%20Process/GP.ipynb](notebooks/dev-history/Gaussian%20Process/GP.ipynb)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
```bash
# Clone the repository
git clone https://github.com/AshutoshKumar1007/ElegantML.git
cd ElegantML

# Create virtual environment
python -m venv venv
# Windows PowerShell
venv\Scripts\Activate.ps1
# macOS/Linux
source venv/bin/activate

# Install core dependencies if needed
pip install numpy pandas matplotlib
```

## 📋 Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0
- Pandas >= 1.2.0

## 🗺️ Roadmap

- [ ] Add more activation functions (GELU, Swish, etc.)
- [ ] Implement Convolutional layers
- [ ] Add Recurrent layers (RNN, LSTM, GRU)
- [ ] Random Forest implementation
- [ ] Support Vector Machines (SVM)
- [ ] Naive Bayes classifiers
- [ ] Principal Component Analysis (PCA)
- [ ] More comprehensive documentation
- [ ] Tutorial notebooks
- [ ] Performance benchmarks

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

ElegantML is built from scratch with inspiration from:
- scikit-learn for API design
- PyTorch for neural network architecture
- Various academic papers and textbooks on machine learning

## 📞 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Made with ❤️ for ML education and transparency**
