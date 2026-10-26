import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Load Adult Income dataset
data = fetch_openml("adult", version=2, as_frame=True)
df = data.frame.dropna(subset=["class"])

# 2. Separate inputs and target
X = df.drop("class", axis=1)
y = df["class"].astype(str).str.strip()

# 3. Identify numerical and categorical columns
num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(exclude="number").columns

# 4. Preprocess the data
num_pipe = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

cat_pipe = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Create models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        criterion="entropy", max_depth=10, random_state=42
    )
}

# 7. Train and compare
for name, model in models.items():
    clf = make_pipeline(preprocess, model)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(
        y_test, pred, pos_label=">50K", zero_division=0))
    print("Recall:", recall_score(
        y_test, pred, pos_label=">50K", zero_division=0))
    print("F1 Score:", f1_score(
        y_test, pred, pos_label=">50K", zero_division=0))