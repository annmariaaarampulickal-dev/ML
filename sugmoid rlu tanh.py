import time
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# 1. Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. Normalize and use smaller subsets
X_train = X_train[:10000].astype("float32") / 255.0
y_train = y_train[:10000]
X_test = X_test[:2000].astype("float32") / 255.0
y_test = y_test[:2000]

# 3. Convert labels to one-hot form
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 4. Compare activation functions
for act in ["sigmoid", "relu", "tanh"]:
    model = Sequential([
        Input(shape=(28, 28)),
        Flatten(),
        Dense(64, activation=act),
        Dense(10, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    start = time.time()
    model.fit(X_train, y_train, epochs=3, batch_size=128, verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    print(act, "Accuracy:", acc)
    print(act, "Training time:", round(time.time() - start, 2), "seconds")