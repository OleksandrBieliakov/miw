from keras import layers
from keras import models
from keras.datasets import cifar10
from keras.utils import to_categorical
import numpy as np


def get_model(n):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    if n == 2:
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
    if n == 3:
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()
    return model


(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images.reshape((50000, 32, 32, 3))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 32, 32, 3))
test_images = test_images.astype('float32') / 255

all_images = np.concatenate((train_images, test_images), axis=0)
all_labels = np.concatenate((train_labels, test_labels), axis=0)
print('all images shape', all_images.shape)
print('all labels shape', all_labels.shape)
train_images, test_images = np.split(all_images, [int(len(all_images)/10*3)])
print('train images shape', train_images.shape)
print('train labels shape', test_images.shape)
train_labels, test_labels = np.split(all_labels, [int(len(all_labels)/10*3)])
print('test images shape', train_labels.shape)
print('test labels shape', test_labels.shape)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = get_model(3)
# optimizer='rmsprop'
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc = ', test_acc)
