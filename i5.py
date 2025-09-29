# Task 5: Decision Trees and Random Forests (Simplified Version)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset (update the path)
df = pd.read_csv(r"C:\Users\Admin\Downloads\intern 5\heart.csv")

# Features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Decision Tree ---
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

# Visualize tree
plt.figure(figsize=(15, 8))
plot_tree(dt, feature_names=X.columns, class_names=["No Disease", "Disease"], filled=True)
plt.show()

# Control overfitting with max_depth
dt2 = DecisionTreeClassifier(max_depth=4, random_state=42)
dt2.fit(X_train, y_train)
print("Decision Tree (max_depth=4) Accuracy:", accuracy_score(y_test, dt2.predict(X_test)))

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# Feature Importances
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=importances.index)
plt.title("Feature Importances (Random Forest)")
plt.show()

# Cross-validation
cv_dt = cross_val_score(dt, X, y, cv=5).mean()
cv_rf = cross_val_score(rf, X, y, cv=5).mean()
print("Cross-validation Decision Tree Accuracy:", cv_dt)
print("Cross-validation Random Forest Accuracy:", cv_rf)