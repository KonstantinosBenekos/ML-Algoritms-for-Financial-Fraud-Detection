# Import Libraries
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report, confusion_matrix

# Data
data = pd.read_csv(r"C:\Users\kwsta\Desktop\Master_thesis\CC_advanced\archive (1)\creditcard.csv")
data.dropna(axis=0, inplace=True)
#data.drop_duplicates(inplace=True)
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

# Split set
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
    Train SVM models on each subset and aggregate predictions.
    
    Parameters:
    subsets (list of tuples): List of (X_subset, y_subset) for training.
    X_test (np.ndarray): Scaled test features.
    y_test (pd.Series): Test target variable.
    
    Returns:
    None
    """
    
    # Initialize lists to store predictions from each model
    all_predictions = np.zeros((len(X_test), len(subsets)))
    
    for i, (X_subset, y_subset) in enumerate(subsets):
        # Train SVM on this subset
        svm_classifier = SVC(kernel="linear", probability=True, random_state=42)
        calibrated_svm = CalibratedClassifierCV(svm_classifier)
        calibrated_svm.fit(X_subset, y_subset)
        
        # Predict on the test set
        all_predictions[:, i] = calibrated_svm.predict(X_test)
    
    # Aggregate predictions by majority voting
    final_predictions = [np.bincount(predictions.astype(int)).argmax() for predictions in all_predictions]
    
    # Evaluate performance
    precision = precision_score(y_test, final_predictions)
    recall = recall_score(y_test, final_predictions)
    f1 = f1_score(y_test, final_predictions)
    accuracy = accuracy_score(y_test, final_predictions)
    
    print("\nAggregated Performance Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print confusion matrix and classification report
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, final_predictions))
    print("Classification Report:")
    print(classification_report(y_test, final_predictions))

# Aggregate and evaluate models
AggregateModels(subsets, X_test_scaled, Y_test)


