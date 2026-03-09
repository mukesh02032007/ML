# Step 1: Import required libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
# Step 2: Example training documents and their labels
documents = [
    "doctor medicine",      # Class 0 → medical
    "doctor patient",       # Class 0 → medical
    "computer graphics",    # Class 1 → graphics
    "image graphics"        # Class 1 → graphics
]
labels = [0, 0, 1, 1]       # 0 = medical, 1 = graphics
# Step 3: Convert text documents into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
# Step 4: Train Naive Bayes model
model = MultinomialNB()
model.fit(X, labels)
# Step 5: Test with new documents
test_docs = [
    "doctor medicine patient",
    "graphics image computer"
]
X_test = vectorizer.transform(test_docs)
predicted = model.predict(X_test)
# Step 6: Print predicted classes
print("Predicted classes:", predicted)
# Step 7: Evaluate model performance
test_labels = [0, 1]   # Actual labels for test documents
print("Accuracy:", accuracy_score(test_labels, predicted))
print("Classification Report:")
print(classification_report(test_labels, predicted))
