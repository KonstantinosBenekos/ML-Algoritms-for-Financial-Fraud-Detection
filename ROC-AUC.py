# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, roc_auc_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

# Load Data
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

# Split set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Create Subsets Function
def CreateSubsets(X, y, n_subsets=10, random_state=None):
    """
    Create multiple balanced subsets of the data with unique majority class samples.
    """
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

# Aggregate Models Function for Random Forest
def AggregateModels_rf(subsets, X_test, y_test):
    all_probabilities = np.zeros((len(X_test), len(subsets)))
    for i, (X_subset, y_subset) in enumerate(subsets):
        rf_classifier = RandomForestClassifier(random_state=42)
        rf_classifier.fit(X_subset, y_subset)
        all_probabilities[:, i] = rf_classifier.predict_proba(X_test)[:, 1]
    final_probabilities = np.mean(all_probabilities, axis=1)
    roc_auc = roc_auc_score(y_test, final_probabilities)
    print(f"AUC (ROC) Random Forest: {roc_auc:.4f}")
    return final_probabilities, roc_auc

# Function to train and aggregate Logistic Regression models
def AggregateModels_logit(subsets, X_test, y_test):
    all_probabilities = np.zeros((len(X_test), len(subsets)))
    for i, (X_subset, y_subset) in enumerate(subsets):
        log_reg = LogisticRegression(max_iter=1000, random_state=42)
        log_reg.fit(X_subset, y_subset)
        all_probabilities[:, i] = log_reg.predict_proba(X_test)[:, 1]
    final_probabilities = np.mean(all_probabilities, axis=1)
    roc_auc = roc_auc_score(y_test, final_probabilities)
    print(f"AUC (ROC) Logistic Regression: {roc_auc:.4f}")
    return final_probabilities, roc_auc

# Function to train and aggregate SVM models
def AggregateModels_SVM(subsets, X_test, y_test):
    all_probabilities = np.zeros((len(X_test), len(subsets)))
    for i, (X_subset, y_subset) in enumerate(subsets):
        svm_classifier = SVC(kernel="linear", probability=True, random_state=42)
        calibrated_svm = CalibratedClassifierCV(svm_classifier)
        calibrated_svm.fit(X_subset, y_subset)
        all_probabilities[:, i] = calibrated_svm.predict_proba(X_test)[:, 1]
    final_probabilities = np.mean(all_probabilities, axis=1)
    roc_auc = roc_auc_score(y_test, final_probabilities)
    print(f"AUC (ROC) SVM: {roc_auc:.4f}")
    return final_probabilities, roc_auc

# Create Subsets
subsets = CreateSubsets(X_train, Y_train, n_subsets=10, random_state=42)

# Scale the data for Logistic Regression and SVM
scaler = StandardScaler()
scaled_subsets = [(pd.DataFrame(scaler.fit_transform(X_subset), columns=X.columns), y_subset) for X_subset, y_subset in subsets]
X_test_scaled = scaler.transform(X_test)

# Aggregate and evaluate models
rf_probabilities, rf_auc = AggregateModels_rf(subsets, X_test, Y_test)
logit_probabilities, logit_auc = AggregateModels_logit(scaled_subsets, X_test_scaled, Y_test)
svm_probabilities, svm_auc = AggregateModels_SVM(scaled_subsets, X_test_scaled, Y_test)

# ROC Curve Data
rf_fpr, rf_tpr, _ = roc_curve(Y_test, rf_probabilities)
logit_fpr, logit_tpr, _ = roc_curve(Y_test, logit_probabilities)
svm_fpr, svm_tpr, _ = roc_curve(Y_test, svm_probabilities)
random_fpr = [0, 1]
random_tpr = [0, 1]

# Plot ROC Curves
plt.figure(figsize=(10, 8))
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.4f})')
plt.plot(logit_fpr, logit_tpr, label=f'Logistic Regression (AUC = {logit_auc:.4f})')
plt.plot(svm_fpr, svm_tpr, label=f'SVM (AUC = {svm_auc:.4f})')
plt.plot(random_fpr, random_tpr, linestyle='--', label='Random (AUC = 0.5000)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
