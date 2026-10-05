import pandas as pd
import math

# Load the dataset
file_path = "datasets/house_price_regression_dataset.csv"
df = pd.read_csv(file_path)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate features and target
X = df.drop("House_Price", axis=1)
y = df["House_Price"]

# Split: 80% training, 20% testing
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
y_train = y.iloc[:split_index]

X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

# Initialize model parameters
w = pd.Series(
    [0.0] * X_train.shape[1],
    index=X_train.columns
)
b = 0

# Set learning rate
learning_rate = 0.00000001

# Set regularization strength
lambda_reg = 1

# Number of iterations
iterations = 50000

# Store loss from each iteration
loss_history = []

# Gradient Descent
for i in range(iterations):

    # Make predictions
    predictions = X_train.dot(w) + b

    # Calculate errors
    errors = predictions - y_train

    # Calculate MSE
    mse = (errors ** 2).mean()

    # Calculate L2 regularization penalty
    regularization = lambda_reg * (w ** 2).sum()

    # Total training loss
    loss = mse + regularization

    # Store loss
    loss_history.append(loss)

    # Calculate gradients
    n = len(y_train)

    # MSE gradient + L2 regularization gradient
    dw = (2 / n) * (X_train.T.dot(errors)) + 2 * lambda_reg * w

    # Bias is not regularized
    db = (2 / n) * errors.sum()

    # Update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # Print progress every 1,000 iterations
    if (i + 1) % 1000 == 0:
        print(
            f"Iteration {i + 1}: "
            f"MSE = {mse}, "
            f"Loss = {loss}, "
            f"w = {w}, "
            f"b = {b}"
        )

# Evaluate on test data
test_predictions = X_test.dot(w) + b

test_errors = test_predictions - y_test

# Test MSE
test_mse = (test_errors ** 2).mean()

# Test RMSE
test_rmse = math.sqrt(test_mse)

print("\nTest MSE:", test_mse)
print("Test RMSE:", test_rmse)

# Print final parameters
print("\nFinal model parameters:")

for feature, weight in zip(X_train.columns, w):
    print(f"{feature}: {weight}")

print(f"Bias: {b}")