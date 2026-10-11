
# TASK 1: DATA PREPROCESSING AND EXPLORATORY DATA ANALYSIS

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# 1. Load the dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = [iris.target_names[i] for i in iris.target]

print("===== TASK 1: DATA PREPROCESSING AND EDA =====")

# 2. Display first five rows
print("\n1. First Five Rows:")
print(df.head())

# 3. Dataset information
print("\n2. Dataset Information:")
df.info()

# 4. Dataset shape
print("\n3. Dataset Shape:", df.shape)

# 5. Check missing values
print("\n4. Missing Values:")
print(df.isnull().sum())

# Handle missing values if present
for column in df.select_dtypes(include="number").columns:
    if df[column].isnull().any():
        df[column] = df[column].fillna(df[column].median())

# 6. Check and remove duplicate rows
print("\n5. Duplicate Rows:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print("Shape after removing duplicates:", df.shape)

# 7. Statistical summary
print("\n6. Statistical Summary:")
print(df.describe())

# 8. Species distribution
print("\n7. Species Count:")
print(df["species"].value_counts())

# 9. Detect possible outliers using IQR
print("\n8. Outlier Detection:")
numeric_columns = df.select_dtypes(include="number").columns

for column in numeric_columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = ((df[column] < lower) | (df[column] > upper)).sum()
    print(f"{column}: {count} possible outliers")

# 10. Feature distribution plots
df[numeric_columns].hist(figsize=(10, 8), bins=15, edgecolor="black")
plt.suptitle("Distribution of Iris Features")
plt.tight_layout()
plt.show()

# 11. Scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="sepal length (cm)",
    y="petal length (cm)",
    hue="species"
)
plt.title("Sepal Length vs Petal Length")
plt.tight_layout()
plt.show()

# 12. Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    df[numeric_columns].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 13. Final result
print("\n9. FINAL SUMMARY")
print("Dataset: Iris Flower Dataset")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))
print("Missing values remaining:", df.isnull().sum().sum())
print("Task 1 completed successfully!")
