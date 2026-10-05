import pandas as pd
import matplotlib.pyplot as plt


# Load the dataset
file_path = "datasets/house_price_regression_dataset.csv"
df = pd.read_csv(file_path)


# Calculate correlation matrix
correlation_matrix = df.corr(numeric_only=True)


# Print correlation with House Price
print("Correlation with House Price:")
print(correlation_matrix["House_Price"].sort_values(ascending=False))


# Plot correlation matrix
plt.figure(figsize=(10, 8))

plt.imshow(correlation_matrix, cmap="coolwarm", aspect="auto")
plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.title("Correlation Matrix")
plt.tight_layout()

plt.savefig("outputs/correlation_matrix.png", dpi=300)

