import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score
data = pd.read_csv("tennis.csv")
le_outlook = LabelEncoder()
le_temperature = LabelEncoder()
le_humidity = LabelEncoder()
le_wind = LabelEncoder()
le_play = LabelEncoder()
data['Outlook'] = le_outlook.fit_transform(data['Outlook'])
data['Temperature'] = le_temperature.fit_transform(data['Temperature'])
data['Humidity'] = le_humidity.fit_transform(data['Humidity'])
data['Wind'] = le_wind.fit_transform(data['Wind'])
data['Play'] = le_play.fit_transform(data['Play'])
X = data[['Outlook', 'Temperature', 'Humidity', 'Wind']]
y = data['Play']
model = CategoricalNB()
# Train using full dataset
model.fit(X, y)
# Predict on same dataset
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy:{accuracy:.2f}")
