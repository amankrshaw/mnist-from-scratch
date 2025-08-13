# train.py
from neural_network import Model, Layer_Dense, Activation_ReLU, Activation_Softmax, Layer_Dropout
from neural_network import Loss_categoricalcrossentropy, Optimizer_Adam, Accuracy_Categorical,convert
import numpy as np
import pickle


if __name__ == "__main__":
    # ===== TRAINING SECTION =====
    # convert("archive/train-images.idx3-ubyte", "archive/train-labels.idx1-ubyte",
    #         "mnist_train.csv", 60000)
    # convert("archive/t10k-images.idx3-ubyte", "archive/t10k-labels.idx1-ubyte",
    #         "mnist_test.csv", 10000)

    # Load CSV data
    train_data = np.loadtxt('mnist_train.csv', delimiter=',')
    test_data = np.loadtxt('mnist_test.csv', delimiter=',')

    X_train = train_data[:, 1:]
    y_train = train_data[:, 0].astype(int)
    X_test = test_data[:, 1:]
    y_test = test_data[:, 0].astype(int)

    # Normalize
    X_train = (X_train.reshape(X_train.shape[0], -1).astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

    # Create and train model
    model = Model()
    model.add(Layer_Dense(784, 128,weight_regularizer_l2 = 5e-4 ,bias_regularizer_l2 = 5e-4))
    model.add(Layer_Dropout(0.1))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 64))
    model.add(Layer_Dropout(0.1))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(64, 32))
    model.add(Layer_Dropout(0.1))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(32, 10))
    model.add(Activation_Softmax())

    model.set(
        loss=Loss_categoricalcrossentropy(),
        optimizer=Optimizer_Adam(learning_rate=0.001, decay=5e-5),
        accuracy=Accuracy_Categorical()
    )
    model.finalize()

    model.train(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), print_every=100)

    # Save model for both pickle and nnfs formats
    model.save('number_mnist.model')
    with open("mnist_model.pkl", "wb") as f:
        pickle.dump(model, f)

