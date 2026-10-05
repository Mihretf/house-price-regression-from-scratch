import pandas as pd
import matplotlib.pyplot as plt


# Load the dataset
file_path = "datasets/house_price_regression_dataset.csv"
df = pd.read_csv(file_path)

# Basic information
print("Dataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Duplicate rows
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# Basic statistics
print("\nBasic statistics:")
print(df.describe())

# Plot distributions of all numerical columns
df.hist(figsize=(12, 10), bins=20)

plt.tight_layout()
plt.savefig("outputs/data_distributions.png", dpi=300)
