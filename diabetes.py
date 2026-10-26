import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

cols = ["Pregnancies", "Glucose", "BP", "SkinThickness",
        "Insulin", "BMI", "DPF", "Age", "Outcome"]

df = pd.read_csv(url, names=cols)

# 2. Separate features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Model WITHOUT scaling
m1 = LogisticRegression(max_iter=1000)
m1.fit(X_train, y_train)
p1 = m1.predict(X_test)

print("WITHOUT SCALING")
print("Accuracy:", accuracy_score(y_test, p1))
print("Precision:", precision_score(y_test, p1, zero_division=0))
print("Recall:", recall_score(y_test, p1, zero_division=0))
print("F1 Score:", f1_score(y_test, p1, zero_division=0))

# 5. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Model WITH scaling
m2 = LogisticRegression(max_iter=1000)
m2.fit(X_train_scaled, y_train)
p2 = m2.predict(X_test_scaled)

print("\nWITH SCALING")
print("Accuracy:", accuracy_score(y_test, p2))
print("Precision:", precision_score(y_test, p2, zero_division=0))
print("Recall:", recall_score(y_test, p2, zero_division=0))
print("F1 Score:", f1_score(y_test, p2, zero_division=0))