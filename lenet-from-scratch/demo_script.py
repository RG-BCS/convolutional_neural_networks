import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics

from lenet import build_lenet
from mnist_data_loader import load_mnist_data
from utils import plot_training_curves, plot_predictions

def main():
    # Set seed for reproducibility
    seed = 42
    tf.random.set_seed(seed)

    # Hyperparameters
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.01

    # Load data
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_mnist_data()

    # Build model
    model = build_lenet(input_shape=X_train.shape[1:], num_classes=10)

    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[metrics.SparseCategoricalAccuracy()]
    )

    # Evaluate before training
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"LeNet Test Accuracy BEFORE Training: {test_acc * 100:.2f}%")

    # Train
    history = model.fit(
        X_train, y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_data=(X_valid, y_valid),
        verbose=2
    )

    # Evaluate after training
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"LeNet Test Accuracy AFTER Training: {test_acc * 100:.2f}%")

    # Plot training curves
    plot_training_curves(history)

    # Plot prediction samples
    plot_predictions(model, X_test, y_test, row=2, col=8, figsize=(15, 5))

if __name__ == "__main__":
    main()
