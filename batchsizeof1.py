import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten

# Load in mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Build model
model = keras.models.Sequential([
		Flatten(input_shape=(28,28,1)),
		Dense(28*28, activation='relu'),
		Dense((28*28)/3, activation='relu'),
		Dense(10, activation='softmax')
	])

# Compile model
model.compile(keras.optimizers.Adam(lr=1e-4), loss='categorical_crossentropy')

model.fit(x=x_train, y=y_train, epochs=10, validation_data=(x_test,y_test), batch_size=64)