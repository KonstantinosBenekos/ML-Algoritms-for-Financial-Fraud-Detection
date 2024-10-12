import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix
import numpy as np

# Data
data = pd.read_csv(r"C:\Users\kwsta\Desktop\Master_thesis\CC_advanced\archive (1)\creditcard.csv")
data.dropna(axis=0, inplace=True)

# Handle outliers
def handle_outliers(df, column, method='IQR', threshold=1.5):
    if method == 'IQR':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

data = handle_outliers(data, 'Amount', method='IQR')

# Dependent/Independent variables
X = data.drop("Class", axis=1)
Y = data["Class"]

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Calculate and print class distribution
def print_class_distribution(y, label):
    print(f"Class distribution in {label}:")
    print(y.value_counts(normalize=True) * 100)
    print()

print_class_distribution(Y, "Original Data")
print_class_distribution(Y_train, "Training Data")
print_class_distribution(Y_test, "Test Data")

# Create Subsets Function
def CreateSubsets(X, y, n_subsets=10, random_state=None):
    X_minority = X[y == 1]
    X_majority = X[y == 0]
    y_minority = y[y == 1]
    y_majority = y[y == 0]

    subsets = []
    majority_indices = y_majority.index.tolist()

    for i in range(n_subsets):
        if random_state is not None:
            np.random.seed(random_state + i)
        shuffled_indices = np.random.permutation(majority_indices)
        selected_indices = shuffled_indices[:len(X_minority)]

        X_majority_sample = X_majority.loc[selected_indices]
        y_majority_sample = y.loc[selected_indices]

        X_subset = pd.concat([X_minority, X_majority_sample])
        y_subset = pd.concat([y_minority, y_majority_sample])
        
        subsets.append((X_subset, y_subset))
    
    return subsets

# Create balanced subsets
subsets = CreateSubsets(X_train, Y_train, n_subsets=10, random_state=42)

# Scale the data
scaler = StandardScaler()
for i, (X_subset, y_subset) in enumerate(subsets):
    X_subset_scaled = scaler.fit_transform(X_subset)
    subsets[i] = (pd.DataFrame(X_subset_scaled, columns=X.columns), y_subset)

X_test_scaled = scaler.transform(X_test)

# Function to train and aggregate models
def AggregateModels(subsets, X_test, y_test):
    """
    Train Logistic Regression models on each subset and aggregate predictions.
    
    Parameters:
    subsets (list of tuples): List of (X_subset, y_subset) for training.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): Test target variable.
    
    Returns:
    avg_coefficients (np.ndarray): Averaged coefficients from all models.
    """
    
    all_predictions = np.zeros((len(X_test), len(subsets)))
    all_coefficients = np.zeros((len(subsets), X_test.shape[1]))
    
    for i, (X_subset, y_subset) in enumerate(subsets):
        log_reg = LogisticRegression(max_iter=1000, random_state=42)
        log_reg.fit(X_subset, y_subset)

        all_coefficients[i, :] = log_reg.coef_[0]
        all_predictions[:, i] = log_reg.predict(X_test)

    final_predictions = [np.bincount(predictions.astype(int)).argmax() for predictions in all_predictions]
    
    precision = precision_score(y_test, final_predictions)
    recall = recall_score(y_test, final_predictions)
    f1 = f1_score(y_test, final_predictions)
    accuracy = accuracy_score(y_test, final_predictions)
    
    print("\nAggregated Performance Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, final_predictions))
    print("Classification Report:")
    print(classification_report(y_test, final_predictions))

    avg_coefficients = np.mean(all_coefficients, axis=0)
    
    return avg_coefficients

# Train and aggregate models on subsets
avg_coefficients_subsets = AggregateModels(subsets, X_test_scaled, Y_test)

# Train and evaluate model on the raw data without subsets
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg_raw = LogisticRegression(max_iter=1000, random_state=42)
log_reg_raw.fit(X_train_scaled, Y_train)

# Predict and evaluate on test data
y_pred_raw = log_reg_raw.predict(X_test_scaled)
precision_raw = precision_score(Y_test, y_pred_raw)
recall_raw = recall_score(Y_test, y_pred_raw)
f1_raw = f1_score(Y_test, y_pred_raw)
accuracy_raw = accuracy_score(Y_test, y_pred_raw)

print("\nPerformance Metrics on Raw Data:")
print(f"Precision: {precision_raw:.4f}")
print(f"Recall: {recall_raw:.4f}")
print(f"F1 Score: {f1_raw:.4f}")
print(f"Accuracy: {accuracy_raw:.4f}")

print("Confusion Matrix (Raw Data):")
print(confusion_matrix(Y_test, y_pred_raw))
print("Classification Report (Raw Data):")
print(classification_report(Y_test, y_pred_raw))

# Retrieve and display feature weights
feature_names = X.columns
coefficients_raw = log_reg_raw.coef_[0]

# Create DataFrame to show feature names and their corresponding weights for both methods
feature_weights_subsets = pd.DataFrame({'Feature': feature_names, 'Weight (Subsets)': avg_coefficients_subsets})
feature_weights_raw = pd.DataFrame({'Feature': feature_names, 'Weight (Raw Data)': coefficients_raw})

# Merge both DataFrames for comparison
feature_weights_combined = pd.merge(feature_weights_subsets, feature_weights_raw, on='Feature')
feature_weights_combined = feature_weights_combined.sort_values(by='Weight (Subsets)', ascending=False)

print("\nFeature Weights Comparison:")
print(feature_weights_combined)

