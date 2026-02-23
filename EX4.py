import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

# Step 1: Read CSV file
data = pd.read_csv("tennis.csv")

# Step 2: Encode categorical columns
le = LabelEncoder()

for column in data.columns:
    data[column] = le.fit_transform(data[column])

# Step 3: Split features and target
X = data.drop("play", axis=1)
y = data["play"]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 5: Train Naive Bayes Model
model = CategoricalNB()
model.fit(X_train, y_train)

# Step 6: Predict on test data
y_pred = model.predict(X_test)

# Step 7: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 8: Predict for outlook = sunny
# We need full feature input, so we give sample values
# (sunny, hot, high, weak)

sample = pd.DataFrame([["sunny", "hot", "high", "weak"]],
                      columns=["outlook", "temperature", "humidity", "wind"])

# Encode sample
for column in sample.columns:
    sample[column] = le.fit_transform(sample[column])

prediction = model.predict(sample)

if prediction[0] == 1:
    print("Prediction for sunny: Player WILL play")
else:
    print("Prediction for sunny: Player WILL NOT play")