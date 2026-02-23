import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("tennis.csv")

# Create encoders
le_outlook = LabelEncoder()
le_temperature = LabelEncoder()
le_humidity = LabelEncoder()
le_wind = LabelEncoder()
le_play = LabelEncoder()

# Encode data
data['Outlook'] = le_outlook.fit_transform(data['Outlook'])
data['Temperature'] = le_temperature.fit_transform(data['Temperature'])
data['Humidity'] = le_humidity.fit_transform(data['Humidity'])
data['Wind'] = le_wind.fit_transform(data['Wind'])
data['Play'] = le_play.fit_transform(data['Play'])

# Features and target
X = data[['Outlook', 'Temperature', 'Humidity', 'Wind']]
y = data['Play']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Model
model = CategoricalNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy (ONLY OUTPUT)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)