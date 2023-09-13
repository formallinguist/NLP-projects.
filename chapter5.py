import numpy as np

example_input = [1, .2, .1, .05, .2]
example_weights = [.2, .12, .4, .6, .90]

input_vector = np.array(example_input)
weights = np.array(example_weights)
bias_weight = .2

activation_level = np.dot(input_vector, weights) +\
        (bias_weight * 1)

print(activation_level)


threshold = 0.5
if activation_level >= threshold:
    perceptron_output = 1
else:
    perceptron_output = 0

print(perceptron_output)

expected_output = 0
new_weights = []
for i, x in enumerate(example_input):
    new_weights.append(weights[i] + (expected_output - \
            perceptron_output) * x)

weights = np.array(new_weights)

print(example_weights)
print(weights)

#OR problem setup.

sample_data = [[0, 0],
               [0, 1],
               [1, 0],
               [1, 1]]

expected_results =[0,
                   1,
                   1,
                   1]

activation_threshold = 0.5

from random import random
import numpy as np

weights = np.random.random(2)/1000

print("weights",weights)

bias_weight = np.random.random() / 1000
print("BiasWeight", bias_weight)


#Perceptron random guessing.

for idx, sample in enumerate(sample_data):
    input_vector = np.array(sample)
    activation_level = np.dot(input_vector, weights) + \
            (bias_weight * 1)
    if activation_level > activation_threshold:
        perceptron_output = 1
    else:
        perceptron_output = 0

    print('Predicted {}'.format(perceptron_output))
    print('Expected: {}'.format(expected_results[idx]))

    print()


#Perceptron learning.

for iteration_num in range(5):
    correct_answers = 0
    for idx, sample in enumerate(sample_data):
        input_vector = np.array(sample)
        weights = np.array(weights)
        activation_level = np.dot(input_vector, weights) +\
                (bias_weight * 1)
        if activation_level > activation_threshold:
            perceptron_output = 1
        else:
            perceptron_output = 0

        if perceptron_output == expected_results[idx]:
            correct_answers += 1

        new_weights = []
        for i, x in enumerate(sample):
            new_weights.append(weights[i] + (expected_results[idx] -\
                    perceptron_output) *  x)
        bias_weight = bias_weight + ((expected_results[idx] -\
                perceptron_output) * 1)
        weights = np.array(new_weights)
    print('{} correct answers out  of 4, for iterations {}'\
            .format(correct_answers, iteration_num))

# XOR With Keras network.
#pip install Tensorflow
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

x_train = np.array([[0, 0],
               [0, 1],
               [1, 0],
               [1, 1]])

y_train = np.array([0,
                   1,
                   1,
                   1])


model = Sequential()
num_neurons = 10
model.add(Dense(num_neurons, input_dim = 2))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.summary()

sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

print(model.predict(x_train))

#Fit the model to the XOR training set.

model.fit(x_train, y_train, epochs=100)

#print(model.predict_classes(x_train))

print(model.predict(x_train))


# Save the created model.
import h5py
model_structure = model.to_json()

with open("basic_model.json","w") as json_file:
    json_file.write(model_structure)

model.save_weights("basic_weights.h5")

