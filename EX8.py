from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# Load dataset
data = load_iris()
X = data.data
y = data.target
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# Create KNN model
knn = KNeighborsClassifier(n_neighbors=1)
# Train model
knn.fit(X_train, y_train)
# Predict
predictions = knn.predict(X_test)
# Print results
for i in range(len(X_test)):
    print("Actual:", data.target_names[y_test[i]],
          "Predicted:", data.target_names[predictions[i]])
# Accuracy
print("Accuracy:", knn.score(X_test, y_test))