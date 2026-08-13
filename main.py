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

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Create and train the Perceptron model
model = Perceptron(random_state=42)
model.fit(x_train, y_train)

# Evaluate the model
accuracy = model.score(x_test, y_test)
print(f"Accuracy: {accuracy:.2f}")


