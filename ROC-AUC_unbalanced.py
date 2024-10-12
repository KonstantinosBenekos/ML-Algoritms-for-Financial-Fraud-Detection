import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, roc_auc_score, classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
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

# Subset for SVM
X_train_svm, Y_train_svm = resample(X_train, Y_train, n_samples=10000, random_state=42, stratify=Y_train)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_svm_scaled = scaler.fit_transform(X_train_svm)

# Define models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Support Vector Machines": CalibratedClassifierCV(SVC(kernel='linear', probability=True, random_state=42))
}

# Train and evaluate models
results = {}
for name, model in models.items():
    if name == "Support Vector Machines":
        model.fit(X_train_svm_scaled, Y_train_svm)
        y_pred = model.predict(X_test_scaled)
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    elif name == "Logistic Regression":
        model.fit(X_train_scaled, Y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Calculate performance metrics
    precision = precision_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)
    accuracy = accuracy_score(Y_test, y_pred)
    roc_auc = roc_auc_score(Y_test, y_pred_prob)

    # Store results
    results[name] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(Y_test, y_pred),
        "classification_report": classification_report(Y_test, y_pred)
    }

# Print performance metrics for each model
for name, metrics in results.items():
    print(f"\nPerformance Metrics for {name}:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC (ROC): {metrics['roc_auc']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print("Classification Report:")
    print(metrics["classification_report"])

plt.figure(figsize=(10, 8))

# Plot each model's ROC AUC curve
for name, model in models.items():
    if name == "Support Vector Machines":
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    elif name == "Logistic Regression":
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {results[name]["roc_auc"]:.4f})')

# Diagonal reference line
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

# Adjust limits to create space between the curves and axes
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])

# Labels, title, and legend
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Add grid
plt.grid(True)

# Display the plot
plt.show()


