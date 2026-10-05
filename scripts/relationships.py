import pandas as pd
import matplotlib.pyplot as plt

file_path = "datasets/house_price_regression_dataset.csv"
df = pd.read_csv(file_path)

features = [
    "Square_Footage",
    "Num_Bedrooms",
    "Num_Bathrooms",
    "Year_Built",
    "Lot_Size",
    "Garage_Size",
    "Neighborhood_Quality"
]

for feature in features:
    plt.figure(figsize=(7, 5))

    plt.scatter(df[feature], df["House_Price"])

    plt.xlabel(feature)
    plt.ylabel("House Price")
    plt.title(f"{feature} vs House Price")

    plt.tight_layout()
    plt.savefig(f"outputs/{feature}_vs_price.png", dpi=300)
    plt.close()