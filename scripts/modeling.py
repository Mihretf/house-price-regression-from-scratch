import pandas as pd


# Load the dataset
file_path = "datasets/house_price_regression_dataset.csv"
df = pd.read_csv(file_path)


# Separate features and target
X = df.drop("House_Price", axis=1)
y = df["House_Price"]


# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Recreate X and y after shuffling
X = df.drop("House_Price", axis=1)
y = df["House_Price"]


# Split: 80% training, 20% testing
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


# Check the sizes
# print("Original dataset:", df.shape)
# print("Training features:", X_train.shape)
# print("Testing features:", X_test.shape)
# print("Training target:", y_train.shape)
# print("Testing target:", y_test.shape)

# Select Square_Footage as our feature
# Use all features
X_train_features = X_train

# Initialize model parameters
w = [0] * X_train_features.shape[1]
b = 0

# Make predictions
predictions = X_train_features.dot(w) + b

print("\nInitial predictions:")
print(predictions.head())

