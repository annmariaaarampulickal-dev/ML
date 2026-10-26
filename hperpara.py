from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# 1. Load dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# 2. Normalize and select subsets
X_train = X_train[:10000].astype("float32") / 255.0
y_train = y_train[:10000]
X_test = X_test[:2000].astype("float32") / 255.0
y_test = y_test[:2000]

# 3. Try different hyperparameters
settings = [
    (0.001, 32, 3),
    (0.01, 64, 5),
    (0.0001, 128, 5)
]

for lr, batch, epochs in settings:
    # Create model
    model = Sequential([
        Input(shape=(28, 28)),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax")
    ])

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Train and evaluate
    model.fit(X_train, y_train, epochs=epochs,
              batch_size=batch, verbose=0)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    print("Learning rate:", lr)
    print("Batch size:", batch, "Epochs:", epochs)
    print("Accuracy:", acc)