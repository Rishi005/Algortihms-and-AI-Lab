import joblib
import mnist_loader
import architecture

MINI_BATCH_SIZE = 10
EPOCHS = 30
ETA = 3.0
HIDDEN_NEURONS = 30
SIZES = [784, HIDDEN_NEURONS, 10]

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = architecture.Network(SIZES)
net.sgd(training_data=training_data, epochs=EPOCHS, mini_batch_size=MINI_BATCH_SIZE, eta=ETA, test_data=test_data)

joblib.dump(net, "trained_model.pkl")

