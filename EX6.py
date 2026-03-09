# COVID Diagnosis using Naive Bayes Classifier

# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# Step 2: Create WHO-style COVID dataset
data_dict = {
    'Fever': ['Yes','Yes','No','Yes','No','Yes','Yes','No','Yes','No','Yes','Yes','No','No','Yes','Yes','No','No','Yes','No'],
    'Cough': ['Yes','Yes','No','No','Yes','Yes','No','No','Yes','No','Yes','Yes','No','Yes','Yes','No','No','Yes','Yes','No'],
    'Fatigue': ['Yes','No','No','Yes','No','Yes','Yes','No','Yes','No','Yes','Yes','No','No','Yes','Yes','No','No','Yes','No'],
    'Breathing_Difficulty': ['No','No','No','Yes','No','Yes','Yes','No','Yes','No','Yes','Yes','No','No','Yes','Yes','No','No','Yes','No'],
    'Loss_of_Smell': ['Yes','No','No','Yes','No','Yes','Yes','No','Yes','No','Yes','Yes','No','No','Yes','Yes','No','No','Yes','No'],
    'Contact_History': ['Yes','Yes','No','Yes','No','Yes','Yes','No','Yes','No','Yes','Yes','No','No','Yes','Yes','No','No','Yes','No'],
    'Covid_Result': [
        'Positive','Positive','Negative','Positive','Negative',
        'Positive','Positive','Negative','Positive','Negative',
        'Positive','Positive','Negative','Negative','Positive',
        'Positive','Negative','Negative','Positive','Negative'
    ]
}

data = pd.DataFrame(data_dict)

print("Dataset Preview:\n")
print(data.head())

# Step 3: Convert Yes/No values to numbers
encoder = LabelEncoder()

for column in data.columns:
    data[column] = encoder.fit_transform(data[column])

# Visualization 1: Symptom Frequency Graph
symptom_counts = data.drop("Covid_Result", axis=1).sum()

plt.figure(figsize=(8,5))
symptom_counts.plot(kind='bar')
plt.title("Frequency of COVID Symptoms")
plt.xlabel("Symptoms")
plt.ylabel("Count")
plt.show()

# Step 4: Split Features and Target
X = data.drop("Covid_Result", axis=1)
y = data["Covid_Result"]

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 7: Prediction
predictions = model.predict(X_test)

print("\nPredicted Results:", predictions)

# Step 8: Model Evaluation
print("\nAccuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# Visualization 2: Confusion Matrix
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Visualization 3: Prediction Probability Graph
probabilities = model.predict_proba(X_test)

plt.figure(figsize=(6,4))
plt.plot(probabilities[:,0], label="Negative Probability")
plt.plot(probabilities[:,1], label="Positive Probability")

plt.title("Prediction Probability for COVID Classification")
plt.xlabel("Test Sample")
plt.ylabel("Probability")
plt.legend()
plt.show()
