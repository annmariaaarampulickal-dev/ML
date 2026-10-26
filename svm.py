import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.svm import SVC

# 1. Load dataset
iris = load_iris()
X = iris.data[:, [0, 2]]   # Sepal length and petal length
y = (iris.target != 0).astype(int)  # 0 = Setosa, 1 = Non-Setosa

# 2. Create and train Linear SVM
model = SVC(kernel="linear", C=1)
model.fit(X, y)

# 3. Plot data points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.title("Linear SVM - Iris Classification")

# 4. Draw decision boundary and margins
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

import numpy as np
xx = np.linspace(xlim[0], xlim[1], 100)
yy = np.linspace(ylim[0], ylim[1], 100)
YY, XX = np.meshgrid(yy, xx)
grid = np.c_[XX.ravel(), YY.ravel()]

Z = model.decision_function(grid).reshape(XX.shape)

ax.contour(XX, YY, Z, levels=[-1, 0, 1],
           colors="black", linestyles=["--", "-", "--"])
plt.show()

# 5. Display margin-related information
print("Number of support vectors:", len(model.support_vectors_))
print("Support vectors per class:", model.n_support_)