#Import neccesary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import resample


#read/overview data
data=pd.read_csv(r"C:\Users\kwsta\Desktop\Master_thesis\CC_advanced\archive (1)\creditcard.csv")
data.head()
data.tail()
data.info()
data.shape
data.describe()
data.columns
data.dtypes

#Deal with missing and duplicate values
missing_values=data.isnull().any(axis=1)
print("Rows with missing values:")
print (missing_values)
#duplicate_rows=data[data.duplicated()]
#print("Duplicated rows:")
#print(duplicate_rows)

# Deal with missing and duplicate values
missing_values = data.isnull().sum()
print("Total number of missing values in each column:")
print(missing_values)

# Total number of missing values
total_missing_values = missing_values.sum()
print(f"\nTotal number of missing values in the dataset: {total_missing_values}")

# Identify duplicate rows
#duplicate_rows = data[data.duplicated()]

# Total number of duplicate rows
#total_duplicate_rows = duplicate_rows.shape[0]
#print(f"\nTotal number of duplicate rows: {total_duplicate_rows}")
data.dropna(axis=0, inplace=True)
data.drop_duplicates(inplace=True)

# Define a function to detect outliers for a specific column
def detect_outliers(df, column, method='IQR', threshold=1.5):
    if method == 'IQR':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    elif method == 'Z-score':
        z_scores = np.abs(stats.zscore(df[column]))
        outliers = df[z_scores > 3]
    return outliers

# Define a function to handle outliers
def handle_outliers(df, column, method='IQR', threshold=1.5):
    if method == 'IQR':
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        # Calculate IQR
        IQR = Q3 - Q1
        # Define outliers as values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        # Filter out outliers
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    elif method == 'Z-score':
        from scipy import stats
        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(df[column]))
        # Define outliers as those with a Z-score greater than 3
        df = df[z_scores < 3]
    return df

# Handle outliers in the 'Amount' column
data = handle_outliers(data, 'Amount', method='IQR')
# Columns to check for outliers
feature_columns = [f'V{i}' for i in range(1, 29)]

# Initialize a list to store outlier information
outlier_summary_list = []

# Detect outliers for each feature column
for column in feature_columns:
    outliers = detect_outliers(data, column, method='IQR')
    
    # Count the number of outliers by class
    total_outliers = outliers.shape[0]
    fraudulent_outliers = outliers[outliers['Class'] == 1].shape[0]
    non_fraudulent_outliers = outliers[outliers['Class'] == 0].shape[0]
    
    # Append summary to list
    outlier_summary_list.append({
        'Feature': column,
        'Total Outliers': total_outliers,
        'Fraudulent Outliers': fraudulent_outliers,
        'Non-Fraudulent Outliers': non_fraudulent_outliers
    })

# Convert the list to a DataFrame
outlier_summary = pd.DataFrame(outlier_summary_list)

# Print outlier summary
print("Outlier Summary:")
print(outlier_summary)

# Calculate the number of fraudulent and non-fraudulent cases
Fraudulent_cases = data["Class"].sum()
Non_Fraudulent_cases = len(data["Class"]) - Fraudulent_cases

# Print the results
print(f"Number of Fraudulent Cases: {Fraudulent_cases}")
print(f"Number of Non-Fraudulent Cases: {Non_Fraudulent_cases}")

categories = ["Non-Fraudulent", "Fraudulent"]
Fraud_NO_Fraud = [Non_Fraudulent_cases, Fraudulent_cases]

plt.pie(Fraud_NO_Fraud, labels=categories, colors=['blue', 'red'], autopct='%1.1f%%')
plt.title("Fraudulent vs Non-Fraudulent Cases")
plt.show()
# Calculate descriptive statistics for the "Amount" attribute by "Class"
amount_stats_by_class = data.groupby('Class')['Amount'].describe()
print(amount_stats_by_class)



# List of features to analyze
features = [f"V{i}" for i in range(1, 29)]
"""
for feature in features:
    # Calculate the correlation
    correlations_df = data[[feature, "Class"]]
    correlation = correlations_df[feature].corr(correlations_df["Class"])
    print(f"Correlation Between Feature {feature} and Class: {correlation}")
    
    # Plot the density plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=data[data["Class"] == 0][feature], label="Non-Fraudulent", fill=True)
    sns.kdeplot(data=data[data["Class"] == 1][feature], label="Fraudulent", fill=True)
    plt.title(f"Density Plot of Feature ({feature}) by Fraud Class")
    plt.xlabel(f"Feature ({feature})")
    plt.ylabel("Density")
    plt.legend()
    plt.show()
""" 

# Create a boxplot for the Amount attribute by Class
plt.figure(figsize=(10, 6))
sns.boxplot(x="Class", y="Amount", data=data)
plt.title("Boxplot of Transaction Amount by Fraud Class")
plt.xlabel("Fraud Class")
plt.ylabel("Transaction Amount")
plt.xticks([0, 1], ['Non-Fraudulent', 'Fraudulent'])
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(data=data[data["Class"] == 0]["Amount"], label="Non-Fraudulent", fill=True, color='blue')
sns.kdeplot(data=data[data["Class"] == 1]["Amount"], label="Fraudulent", fill=True, color='red')
plt.title("Density Plot of Transaction Amount by Fraud Class")
plt.xlabel("Transaction Amount")
plt.ylabel("Density")
plt.legend()
plt.show()

# Prepare data for training
X = data.drop('Class', axis=1)
y = data['Class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def CreateSubsets(X, y, n_subsets=10, random_state=None):
    """
    Create multiple balanced subsets of the data with unique majority class samples.
    
    Parameters:
    X (pd.DataFrame): Features of the dataset.
    y (pd.Series): Target variable.
    n_subsets (int): Number of subsets to create.
    random_state (int): Seed for reproducibility.
    
    Returns:
    list of tuples: Each tuple contains (X_subset, y_subset) for a balanced subset.
    """
    
    # Separate the minority and majority classes
    X_minority = X[y == 1]
    X_majority = X[y == 0]
    y_minority = y[y == 1]
    y_majority = y[y == 0]

    subsets = []
    majority_indices = y_majority.index.tolist()  # List of indices for majority class

    for i in range(n_subsets):
        # Shuffle majority class indices and select a subset
        if random_state is not None:
            np.random.seed(random_state + i)  # Seed for reproducibility
        shuffled_indices = np.random.permutation(majority_indices)
        selected_indices = shuffled_indices[:len(X_minority)]

        # Sample the selected indices
        X_majority_sample = X_majority.loc[selected_indices]
        y_majority_sample = y.loc[selected_indices]

        # Combine minority class with the sampled majority class
        X_subset = pd.concat([X_minority, X_majority_sample])
        y_subset = pd.concat([y_minority, y_majority_sample])
        
        # Append the balanced subset to the list
        subsets.append((X_subset, y_subset))
    
    return subsets

# Example usage
# X_train and y_train should be defined before calling CreateSubsets
subsets = CreateSubsets(X_train, y_train, n_subsets=10, random_state=42)

# Optionally, inspect the sizes of the subsets
for i, (X_subset, y_subset) in enumerate(subsets):
    print(f"Subset {i+1} - Number of Fraudulent Cases: {y_subset.sum()}")
    print(f"Subset {i+1} - Number of Non-Fraudulent Cases: {len(y_subset) - y_subset.sum()}")

categories = ["Non-Fraudulent", "Fraudulent"]
Fraud_NO_Fraud = [Non_Fraudulent_cases, Fraudulent_cases]

plt.pie(Fraud_NO_Fraud, labels=categories, colors=['blue', 'red'], autopct='%1.1f%%')
plt.title("Fraudulent vs Non-Fraudulent Cases after EasyEnsemble")
plt.show()
