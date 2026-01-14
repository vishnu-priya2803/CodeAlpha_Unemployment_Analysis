import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Load dataset
# ------------------------------
df = pd.read_csv(
    r"C:\Users\vishn\code alpha\Unemployment Analysis with Python\Unemployment in India.csv"
)

# ------------------------------
# Clean column names
# ------------------------------
df.columns = df.columns.str.strip()

# ------------------------------
# Convert 'Date' to datetime
# ------------------------------
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# ------------------------------
# Handle missing values (drop rows with any NaNs)
# ------------------------------
df = df.dropna()

# ------------------------------
# Basic exploration
# ------------------------------
print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# ------------------------------
# Time Series Plot: Unemployment over time
# ------------------------------
plt.figure(figsize=(10,5))
plt.plot(df['Date'], df['Estimated Unemployment Rate (%)'], marker='o')
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.title("Unemployment Rate Over Time in India")
plt.grid(True)
plt.show()

# ------------------------------
# Region-wise Average Unemployment Rate
# ------------------------------
plt.figure(figsize=(12,6))
region_avg_unemp = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values()
sns.barplot(x=region_avg_unemp.index, y=region_avg_unemp.values)
plt.xticks(rotation=90)
plt.ylabel("Average Unemployment Rate (%)")
plt.title("Region-wise Average Unemployment Rate")
plt.show()

# ------------------------------
# Region-wise Average Labour Participation Rate
# ------------------------------
plt.figure(figsize=(12,6))
region_avg_labour = df.groupby('Region')['Estimated Labour Participation Rate (%)'].mean().sort_values()
sns.barplot(x=region_avg_labour.index, y=region_avg_labour.values)
plt.xticks(rotation=90)
plt.ylabel("Average Labour Participation Rate (%)")
plt.title("Region-wise Average Labour Participation Rate")
plt.show()

# ------------------------------
# Correlation Heatmap (numeric columns only)
# ------------------------------
numeric_df = df.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ------------------------------
# COVID-19 Year Analysis (2020)
# ------------------------------
covid_data = df[df['Date'].dt.year == 2020]
plt.figure(figsize=(10,5))
plt.plot(covid_data['Date'], covid_data['Estimated Unemployment Rate (%)'], color='red', marker='o')
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.title("Unemployment Rate During COVID-19 (2020)")
plt.grid(True)
plt.show()

# ------------------------------
# Key Insights
# ------------------------------
print("\nKey Insights:")
print("- Unemployment increased sharply during COVID-19 period.")
print("- Certain regions consistently show higher unemployment.")
print("- Labour participation rate varies significantly across regions.")

# ------------------------------
# Optional: Highest and Lowest Unemployment Region
# ------------------------------
print("\nHighest unemployment region:", region_avg_unemp.idxmax())
print("Lowest unemployment region:", region_avg_unemp.idxmin())

# ------------------------------
# Optional: Unemployment trend by region over time
# ------------------------------
plt.figure(figsize=(12,6))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', hue='Region', data=df)
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.title("Unemployment Rate Trends by Region Over Time")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
