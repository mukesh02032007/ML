import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# -------------------------------
# 1. Load and preprocess data
# -------------------------------
def load_and_preprocess(file_path):
    data = pd.read_csv(file_path)

    # Drop irrelevant columns
    data = data.drop(columns=['customer_id', 'signup_date'], errors='ignore')

    # Convert categorical variables
    data = pd.get_dummies(data, drop_first=True)

    return data


# -------------------------------
# 2. Split data
# -------------------------------
def split_data(data):
    X = data.drop('churned', axis=1)
    y = data['churned']

    return train_test_split(
        X, y, test_size=0.2, random_state=42
    )


# -------------------------------
# 3. Train model
# -------------------------------
def train_model(X_train, y_train):
    model = DecisionTreeClassifier(
        max_depth=3,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model


# -------------------------------
# 4. Evaluate model
# -------------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nAccuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return cm


# -------------------------------
# 5. Plot confusion matrix
# -------------------------------
def plot_confusion_matrix(cm):
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['No Churn', 'Churn'],
        yticklabels=['No Churn', 'Churn']
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


# -------------------------------
# 6. Feature importance
# -------------------------------
def plot_feature_importance(model, X):
    importance = model.feature_importances_
    feature_importance = pd.Series(importance, index=X.columns)

    top_features = feature_importance.sort_values(ascending=False).head(10)

    plt.figure(figsize=(8, 5))
    top_features.plot(kind='bar')
    plt.title("Top 10 Important Features")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()


# -------------------------------
# Main execution
# -------------------------------
def main():
    data = load_and_preprocess("data.csv")

    X_train, X_test, y_train, y_test = split_data(data)

    model = train_model(X_train, y_train)

    cm = evaluate_model(model, X_test, y_test)

    plot_confusion_matrix(cm)

    plot_feature_importance(model, X_train)


if __name__ == "__main__":
    main()
