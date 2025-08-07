 1. Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# To display plots in Jupyter Notebook (optional if you're using Jupyter)
# %matplotlib inline

# 2. Load the dataset
# You can download the Titanic dataset from: https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 3. Summary statistics
print("Basic Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# 4. Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 5. Data Visualization

# Histogram for Age
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'].dropna(), kde=True, bins=30)
plt.title('Age Distribution')
plt.show()

# Boxplot of Age by Survival
plt.figure(figsize=(8, 5))
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Age vs Survived')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Pairplot for selected features
selected_features = ['Survived', 'Age', 'Fare', 'Pclass']
sns.pairplot(df[selected_features].dropna(), hue='Survived')
plt.suptitle("Pairplot of selected features", y=1.02)
plt.show()

# 6. Detecting Skewness
print("\nSkewness of Numerical Features:")
print(df.skew(numeric_only=True))

# 7. Multicollinearity Check
from statsmodels.stats.outliers_influence import variance_inflation_factor

numeric_df = df[['Age', 'Fare', 'Pclass']].dropna()
vif_data = pd.DataFrame()
vif_data["feature"] = numeric_df.columns
vif_data["VIF"] = [variance_inflation_factor(numeric_df.values, i) for i in range(len(numeric_df.columns))]
print("\nVariance Inflation Factor (VIF):")
print(vif_data)

# 8. Plot with Plotly (Interactive)
fig = px.scatter(df, x='Age', y='Fare', color='Survived', title='Fare vs Age colored by Survival')
fig.show()
