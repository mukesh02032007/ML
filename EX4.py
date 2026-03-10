import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = CategoricalNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
