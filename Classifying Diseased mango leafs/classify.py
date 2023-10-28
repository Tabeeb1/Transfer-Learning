from keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow.compat.v2 as tf
import numpy as np
from imageio import imread
from skimage.transform import resize
from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.mobilenet_v2 import decode_predictions
from keras.layers import Dense
from keras import Model


model = MobileNetV2(weights='imagenet')
data = np.empty((4000, 224, 224, 3))
for i in range(500):
    im = imread('Anthracnose/anthracnose (' + str(i + 1) + ').jpg')
    im = preprocess_input(im)
    im = resize(im, output_shape=(224, 224))
    data[i] = im
for i in range(500):
    im = imread('Back/dieback (' + str(i + 1) + ').jpg')
    im = preprocess_input(im)
    im = resize(im, output_shape=(224, 224))
    data[i+500] = im
for i in range(500):
    im = imread('Canker/canker (' + str(i + 1) + ').jpg')
    im = preprocess_input(im)
    im = resize(im, output_shape=(224, 224))
    data[i+1000] = im
for i in range(500):
    im = imread('Healthy/healthy (' + str(i + 1) + ').jpg')
    im = preprocess_input(im)
    im = resize(im, output_shape=(224, 224))
    data[i+1500] = im
for i in range(500):
    im = imread('Midge/midge (' + str(i + 1) + ').jpg')
    im = preprocess_input(im)
    im = resize(im, output_shape=(224, 224))
    data[i+2000] = im
for i in range(500):
    im = imread('Mildew/mildew (' + str(i + 1) + ').jpg')
    im = preprocess_input(im)
    im = resize(im, output_shape=(224, 224))
    data[i+2500] = im
for i in range(500):
    im = imread('Mould/mould (' + str(i + 1) + ').jpg')
    im = preprocess_input(im)
    im = resize(im, output_shape=(224, 224))
    data[i+3000] = im
for i in range(500):
    im = imread('Weevil/weevil (' + str(i + 1) + ').jpg')
    im = preprocess_input(im)
    im = resize(im, output_shape=(224, 224))
    data[i+3500] = im

labels = np.empty(4000, dtype=int)
labels[:500] = 0
labels[500:1000] = 1
labels[1000:1500] = 2
labels[1500:2000] = 3
labels[2000:2500] = 4
labels[2500:3000] = 5
labels[3000:3500] = 6
labels[3500:4000] = 7


predictions = model.predict(data)

for decoded_prediction in decode_predictions(predictions, top =1):
    for name, desc, score in decoded_prediction:
        accuracy = 100 * score
        accuracy = round(accuracy, 2)
        print(f'{desc} - Accuracy is {accuracy}')

leaf_output = Dense(8, activation='softmax')
leaf_output = leaf_output(model.layers[-2].output)
leaf_input = model.input
leaf_model = Model(inputs=leaf_input, outputs=leaf_output)
for layer in leaf_model.layers[:-1]:
    layer.trainable = False

leaf_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

leaf_model.fit(
    x=data,
    y=labels,
    epochs=15,
    verbose=2
)

predictions = leaf_model.predict(data)

training_data = np.empty((3200, 224, 224, 3))
training_data[:400] = data[:400]
training_data[400:800] = data[500:900]
training_data[800:1200] = data[1000:1400]
training_data[1200:1600] = data[1500:1900]
training_data[1600:2000] = data[2000:2400]
training_data[2000:2400] = data[2500:2900]
training_data[2400:2800] = data[3000:3400]
training_data[2800:3200] = data[3500:3900]

training_labels = np.empty(3200)
training_labels[:400] = 0
training_labels[400:800] = 1
training_labels[800:1200] = 2
training_labels[1200:1600] = 3
training_labels[1600:2000] = 4
training_labels[2000:2400] = 5
training_labels[2400:2800] = 6
training_labels[2800:3200] = 7



validation_data = np.empty((800, 224, 224, 30))
validation_data[:100] = data[400:500]
validation_data[100:200] = data[900:1000]
validation_data[200:300] = data[1400:1500]
validation_data[300:400] = data[1900:2000]
validation_data[400:500] = data[2400:2500]
validation_data[500:600] = data[2900:3000]
validation_data[600:700] = data[3400:3500]
validation_data[700:800] = data[3900:4000]


validation_labels = np.empty(800)
validation_labels[:100] = 0
validation_labels[100:200] = 1
validation_labels[200:300] = 2
validation_labels[300:400] = 3
validation_labels[400:500] = 4
validation_labels[500:600] = 5
validation_labels[600:700] = 6
validation_labels[700:800] = 7


leaf_model.fit(
    x=training_data,
    y=training_labels,
    epochs=15,
    verbose=2,
    validation_data=(validation_data, validation_labels)
)
