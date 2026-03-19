import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,confusion_matrix
data = pd.read_csv("02_mini project/data.csv")
'''
print("\nFirst 5 rows:")
print(data.head())
print("\nColumns:")
print(data.columns)
print("\nMissing values:")
print(data.isnull().sum())
'''
data = data.drop(columns=['customer_id','signup_date'])
#print(data.info())
data = pd.get_dummies(data, drop_first=True)
X = data.drop('churned', axis=1)
y = data['churned']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model=DecisionTreeClassifier(max_depth=3)
model.fit(X_train,y_train)
'''plt.figure(figsize=(20,10))
plot_tree(model,feature_names=X.columns,class_names=['No churns','Churn'],filled=True)
plt.savefig("Decision_Tree.png")
plt.show()'''
y_predict=model.predict(X_test)
'''print("Accuracy:",accuracy_score(y_test,y_predict))
print("\nConfusion Matrix:")
print(cm)'''
cm=confusion_matrix(y_test, y_predict)
importance = model.feature_importances_
feature_importance = pd.Series(importance, index=X.columns)
#print("\nTop Important Features:")
#print(feature_importance.sort_values(ascending=False).head(10))

'''plt.figure(figsize=(10,5))
feature_importance.sort_values(ascending=False).head(10).plot(kind='bar')
plt.title("Top 10 Important Features")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.savefig("feature_importance.png")
plt.show()'''
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn','Churn'],
            yticklabels=['No Churn','Churn'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()