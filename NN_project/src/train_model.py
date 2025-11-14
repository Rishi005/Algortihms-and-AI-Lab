
import mnist_loader
import architecture
import joblib

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = architecture.Network([784, 30, 10])
net.SGD(training_data=training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)

joblib.dump(net, "trained_model.pkl")

