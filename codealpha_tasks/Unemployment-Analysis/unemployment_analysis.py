import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("Unemployment in India.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Convert date column
df["Date"] = pd.to_datetime(
    df["Date"].astype(str).str.strip(),
    dayfirst=True,
    errors="coerce"
)

# Display dataset information
print("Dataset Overview")
print(df.head())
print("\nDataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

# Remove rows with missing important values
df = df.dropna(subset=["Date", "Estimated Unemployment Rate (%)"])

# Overall unemployment rate
average_rate = df["Estimated Unemployment Rate (%)"].mean()
print("\nAverage Unemployment Rate:", round(average_rate, 2), "%")

# Monthly unemployment trend
monthly = df.groupby(
    df["Date"].dt.to_period("M")
)["Estimated Unemployment Rate (%)"].mean()

monthly.index = monthly.index.astype(str)

plt.figure(figsize=(12, 5))
monthly.plot(marker="o")
plt.title("Unemployment Rate Over Time in India")
plt.xlabel("Month")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("monthly_unemployment_trend.png")
plt.show()

# Region-wise unemployment rate
region_avg = df.groupby("Region")[
    "Estimated Unemployment Rate (%)"
].mean().sort_values(ascending=False)

print("\nRegion-wise Average Unemployment Rate:")
print(region_avg)

plt.figure(figsize=(12, 6))
region_avg.plot(kind="bar")
plt.title("Average Unemployment Rate by Region")
plt.xlabel("Region")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("region_unemployment.png")
plt.show()

print("\nAnalysis completed successfully!")