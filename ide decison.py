import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. Load dataset
df = pd.read_excel("Online Retail.xlsx")

# 2. Remove invalid transactions
df = df.dropna(subset=["CustomerID"])
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# 3. Calculate spending per customer
df["Amount"] = df["Quantity"] * df["UnitPrice"]

customer = df.groupby("CustomerID").agg(
    TotalSpend=("Amount", "sum"),
    TotalItems=("Quantity", "sum"),
    Transactions=("InvoiceNo", "nunique")
).reset_index()

# 4. Create customer segments using spending
customer["Segment"] = pd.qcut(
    customer["TotalSpend"].rank(method="first"),
    q=3, labels=["Low", "Medium", "High"]
)

# 5. Prepare input and target
X = customer[["TotalSpend", "TotalItems", "Transactions"]]
y = customer["Segment"]

# 6. Train entropy-based decision tree
model = DecisionTreeClassifier(
    criterion="entropy", max_depth=3, random_state=42
)
model.fit(X, y)

# 7. Visualize the tree
plt.figure(figsize=(14, 7))
plot_tree(model, feature_names=X.columns,
          class_names=["Low", "Medium", "High"],
          filled=True, rounded=True)
plt.show()

# 8. Display feature importance
print("Feature Importance:")
for name, importance in zip(X.columns, model.feature_importances_):
    print(name, ":", round(importance, 3))