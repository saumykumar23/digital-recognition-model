import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Print dataset shape
print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

# Show one sample image
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.show()
# Scale pixel values from 0-255 to 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0
# Create a simple neural network
model = Sequential([
    Flatten(input_shape=(28, 28)),       # Convert 28x28 image to 1D
    Dense(128, activation='relu'),       # Hidden layer
    Dense(10, activation='softmax')      # Output layer (10 digits)
])
model.compile(
    optimizer='adam',                    # Optimizer
    loss='sparse_categorical_crossentropy',  # Loss function
    metrics=['accuracy']                 # Evaluation metric
)
# Train the model
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)
# Test accuracy
test_loss, test_acc = model.evaluate(x_test, y_test)
print("\n✅ Test Accuracy:", test_acc)
# Make a prediction on one test image
plt.imshow(x_test[0], cmap='gray')
plt.title(f"Actual Label: {y_test[0]}")
plt.show()

pred = model.predict(x_test[0].reshape(1, 28, 28))
print("Predicted Digit:", pred.argmax())

