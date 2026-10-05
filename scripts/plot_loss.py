import pandas as pd
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

# Use Square_Footage as our feature
X_train_feature = X_train["Square_Footage"]

# Initialize model parameters
w = 0
b = 0

# Set learning rate
learning_rate = 0.00000001

# Number of iterations
iterations = 10000

# Store loss from each iteration
loss_history = []

# Gradient Descent
for i in range(iterations):

    # Make predictions
    predictions = w * X_train_feature + b

    # Calculate errors
    errors = predictions - y_train

    # Calculate MSE
    mse = (errors ** 2).mean()
    loss_history.append(mse)

    # Calculate gradients
    n = len(y_train)

    dw = (2 / n) * (X_train_feature * errors).sum()
    db = (2 / n) * errors.sum()

    # Update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db

# Plot the loss
# Plot the first 500 iterations
# Plot the first 100 iterations
plt.figure(figsize=(10, 6))

plt.plot(range(1, 101), loss_history[:100])

plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")
plt.title("Gradient Descent: Loss Over First 100 Iterations")

plt.tight_layout()
plt.savefig("outputs/loss_curve.png", dpi=300)
plt.show()