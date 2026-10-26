import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Load dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# 2. Flatten and normalize
X_train = X_train.reshape(len(X_train), -1) / 255.0
X_test = X_test.reshape(len(X_test), -1) / 255.0

# 3. Use smaller subsets
X_train, y_train = X_train[:2000], y_train[:2000]
X_test, y_test = X_test[:1000], y_test[:1000]

# 4. Compare kernels
for kernel in ["linear", "poly", "rbf"]:
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(kernel, "Accuracy:", accuracy_score(y_test, pred))