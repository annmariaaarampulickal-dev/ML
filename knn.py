import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# 2. Flatten images and normalize pixels
X_train = X_train.reshape(len(X_train), -1) / 255.0
X_test = X_test.reshape(len(X_test), -1) / 255.0

# 3. Use a smaller training set for faster execution
X_train = X_train[:10000]
y_train = y_train[:10000]

# 4. Try different K values
for k in [1, 3, 5, 7]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("K =", k)
    print("Accuracy:", accuracy_score(y_test, y_pred))