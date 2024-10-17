import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load the datasets
dataset1 = pd.read_csv('dataset1.csv')
dataset2 = pd.read_csv('dataset2.csv')
dataset3 = pd.read_csv('dataset3.csv')

# Merge datasets on 'ID'
data = dataset1.merge(dataset2, on='ID').merge(dataset3, on='ID')

# Data Exploration
print("Data Exploration:")
print("Merged Data Info:")
print(data.info())
print("\nMerged Data Description:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Data Visualization
plt.figure(figsize=(12, 6))
sns.histplot(data['C_we'], kde=True, bins=30)
plt.title('Distribution of Computer Usage on Weekends')
plt.xlabel('Hours')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='gender', y='Optm', data=data)
plt.title('Optimism Scores by Gender')
plt.xlabel('Gender')
plt.ylabel('Optimism Score')
plt.show()

# Correlation Matrix
correlation_matrix = data.corr()
plt.figure(figsize=(24, 18))  # Increased figure size
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.1, xticklabels='auto', yticklabels='auto', cbar_kws={"shrink": .8})  # Set square=True to increase box size
plt.title('Correlation Matrix')
plt.show()

# Feature Engineering
data['total_screen_time_we'] = data[['C_we', 'G_we', 'S_we', 'T_we']].sum(axis=1)
data['total_screen_time_wk'] = data[['C_wk', 'G_wk', 'S_wk', 'T_wk']].sum(axis=1)

# Linear Regression Modelling
X = data[['total_screen_time_we', 'total_screen_time_wk']]
y = data['Optm']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")

# Plotting the results as a bar graph
results_df = pd.DataFrame({'Measured': y_test, 'Predicted': y_pred})
results_df = results_df.reset_index(drop=True)
results_df.head(20).plot(kind='bar', figsize=(12, 6))  # Bar graph for the first 20 samples
plt.xlabel('Sample Index')
plt.ylabel('Optimism Score')
plt.title('Measured vs Predicted Optimism Scores (First 20 Samples)')
plt.show()
