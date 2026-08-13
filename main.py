from sklearn.linear_model import Perceptron  # type: ignore
from sklearn.datasets import make_classification  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

# Generate a synthetic dataset for binary classification with 10 features, 1000 samples
x, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_classes=2,
    random_state=42
)


