import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

# 1. Load Dataset
data = pd.read_excel("Telco_customer_churn.xlsx")
# Clean column names
data.columns = data.columns.str.strip()

# REMOVE USELESS / PROBLEM COLUMNS
data = data.drop(columns=[
    'CustomerID',
    'Country',
    'State',
    'City',
    'Zip Code',
    'Lat Long',
    'Churn Label',
    'Churn Reason',
    'Churn Score'
])

# HANDLE NUMERIC ISSUE
data['Total Charges'] = pd.to_numeric(data['Total Charges'], errors='coerce')
data = data.fillna(0)

# CONVERT CATEGORICAL → NUMERIC
data = pd.get_dummies(data)

# 2. Preprocessing
X = data.drop('Churn Value', axis=1)
y = data['Churn Value']

# 3. Train-Test Split (IMPORTANT)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 4. Model (BEST CHOICE)
model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

# 5. Prediction
y_pred = model.predict(X_test)

# 6. Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\nModel Performance:")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# 7. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn','Churn'],
            yticklabels=['No Churn','Churn'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("cm_output.png")
plt.show()

# 8. Metrics Graph
metrics = ['Accuracy','Precision','Recall','F1']
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(6,4))
plt.bar(metrics, values)
plt.ylim(0,1)

for i, v in enumerate(values):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

plt.title("Model Performance Metrics")
plt.savefig("metrics_output.png")
plt.show()
