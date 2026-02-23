import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
data = pd.read_csv("tennis.csv")

# Step 2: Create separate encoders for each column
le_outlook = LabelEncoder()
le_temperature = LabelEncoder()
le_humidity = LabelEncoder()
le_wind = LabelEncoder()
le_play = LabelEncoder()

# Encode features
data['Outlook'] = le_outlook.fit_transform(data['Outlook'])
data['Temperature'] = le_temperature.fit_transform(data['Temperature'])
data['Humidity'] = le_humidity.fit_transform(data['Humidity'])
data['Wind'] = le_wind.fit_transform(data['Wind'])
data['Play'] = le_play.fit_transform(data['Play'])

# Step 3: Split features and target
X = data[['Outlook', 'Temperature', 'Humidity', 'Wind']]
y = data['Play']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 5: Train Naive Bayes model
model = CategoricalNB()
model.fit(X_train, y_train)

# Step 6: Prediction on test data
y_pred = model.predict(X_test)

# Step 7: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Naive Bayes Classifier:", accuracy)

# Step 8: Predict for Outlook = Sunny
# (Sunny, Hot, High, Weak)

sample = pd.DataFrame({
    'Outlook': [le_outlook.transform(['Sunny'])[0]],
    'Temperature': [le_temperature.transform(['Hot'])[0]],
    'Humidity': [le_humidity.transform(['High'])[0]],
    'Wind': [le_wind.transform(['Weak'])[0]]
})

prediction = model.predict(sample)

result = le_play.inverse_transform(prediction)

print("Prediction for Sunny, Hot, High, Weak:", result[0])