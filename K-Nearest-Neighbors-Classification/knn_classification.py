import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# Load dataset
df = pd.read_csv("iris.csv")

# Features and target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Convert target to numbers if needed
if y.dtype == object:
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Try different K values
k_values = [1, 3, 5, 7, 9]

print("Accuracy for different K values:")
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("K =", k, "Accuracy =", round(acc, 2))

# Final model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# =========================
# Decision Boundary (2 features only)
# =========================
X_2d = X[:, :2]
X_2d = scaler.fit_transform(X_2d)

X_train, X_test, y_train, y_test = train_test_split(
    X_2d, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(('red', 'green', 'blue')))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=ListedColormap(('red', 'green', 'blue')))
plt.title("KNN Decision Boundary (Iris)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()