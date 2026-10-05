import pandas as pd
import math
import matplotlib.pyplot as plt

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

# Model settings
learning_rate = 0.00000001
iterations = 10000
lambda_reg = 100000


# ============================================================
# 1. UNREGULARIZED MODEL
# ============================================================

print("\nTraining unregularized model...")

w = pd.Series(
    [0.0] * X_train.shape[1],
    index=X_train.columns
)

b = 0.0

loss_history_before = []

for i in range(iterations):

    # Make predictions
    predictions = X_train.dot(w) + b

    # Calculate errors
    errors = predictions - y_train

    # Calculate MSE
    mse = (errors ** 2).mean()

    # Store loss
    loss_history_before.append(mse)

    # Calculate gradients
    n = len(y_train)

    dw = (2 / n) * X_train.T.dot(errors)
    db = (2 / n) * errors.sum()

    # Update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # Show progress
    if (i + 1) % 1000 == 0:
        print(f"Unregularized - Iteration {i + 1}/{iterations}")

# Evaluate unregularized model
test_predictions_before = X_test.dot(w) + b

test_errors_before = test_predictions_before - y_test

test_mse_before = (test_errors_before ** 2).mean()

test_rmse_before = math.sqrt(test_mse_before)

# Save parameters
weights_before = w.copy()
bias_before = b


# ============================================================
# 2. REGULARIZED MODEL
# ============================================================

print("\nTraining regularized model...")

w = pd.Series(
    [0.0] * X_train.shape[1],
    index=X_train.columns
)

b = 0.0

loss_history_after = []

for i in range(iterations):

    # Make predictions
    predictions = X_train.dot(w) + b

    # Calculate errors
    errors = predictions - y_train

    # Calculate MSE
    mse = (errors ** 2).mean()

    # Calculate L2 regularization penalty
    regularization = lambda_reg * (w ** 2).sum()

    # Calculate total loss
    loss = mse + regularization

    # Store loss
    loss_history_after.append(loss)

    # Calculate gradients
    n = len(y_train)

    # MSE gradient + L2 regularization gradient
    dw = (
        (2 / n) * X_train.T.dot(errors)
        + 2 * lambda_reg * w
    )

    # Bias is not regularized
    db = (2 / n) * errors.sum()

    # Update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # Show progress
    if (i + 1) % 1000 == 0:
        print(f"Regularized - Iteration {i + 1}/{iterations}")

# Evaluate regularized model
test_predictions_after = X_test.dot(w) + b

test_errors_after = test_predictions_after - y_test

test_mse_after = (test_errors_after ** 2).mean()

test_rmse_after = math.sqrt(test_mse_after)

# Save parameters
weights_after = w.copy()
bias_after = b


# ============================================================
# 3. MODEL COMPARISON
# ============================================================

print("\n========== MODEL COMPARISON ==========")

print("\nUnregularized model:")
print("Test MSE:", test_mse_before)
print("Test RMSE:", test_rmse_before)

print("\nRegularized model:")
print("Lambda:", lambda_reg)
print("Test MSE:", test_mse_after)
print("Test RMSE:", test_rmse_after)


# ============================================================
# 4. WEIGHT COMPARISON
# ============================================================

print("\n========== WEIGHT COMPARISON ==========")

for feature in X_train.columns:
    print(
        f"{feature}: "
        f"Before = {weights_before[feature]:.6f}, "
        f"After = {weights_after[feature]:.6f}"
    )

print(
    f"\nBias: "
    f"Before = {bias_before:.6f}, "
    f"After = {bias_after:.6f}"
)


# ============================================================
# 5. LOSS CURVE
# ============================================================

# ============================================================
# 5. LOSS CURVE — FIRST 20 ITERATIONS
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    loss_history_before[:20],
    label="Before regularization"
)

plt.plot(
    loss_history_after[:20],
    label="After regularization"
)

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss: First 20 Iterations")
plt.legend()
plt.grid()

plt.savefig("outputs/loss_regularization_comparison.png")
plt.show()

# ============================================================
# 6. ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    test_predictions_before,
    label="Before regularization"
)

plt.scatter(
    y_test,
    test_predictions_after,
    label="After regularization"
)

# Perfect prediction line
min_price = min(
    y_test.min(),
    test_predictions_before.min(),
    test_predictions_after.min()
)

max_price = max(
    y_test.max(),
    test_predictions_before.max(),
    test_predictions_after.max()
)

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    linestyle="--",
    label="Perfect prediction"
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted: Before vs After Regularization")
plt.legend()
plt.grid()

plt.savefig("outputs/actual_vs_predicted_comparison.png")

plt.show()