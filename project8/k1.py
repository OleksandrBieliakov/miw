from keras import layers
from keras import models
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

model = models.Sequential()
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(layers.Conv2DTranspose(32, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform'))
model.add(layers.Conv2DTranspose(64, kernel_size=(3,3), activation='relu', kernel_initializer='he_uniform'))
model.add(layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same'))
model.summary()


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(train_images, train_images, epochs = 1, batch_size=64)


# Szum
pure = train_images
pure_test = test_images
noise = np.random.normal(0, 1, pure.shape)
noise_test = np.random.normal(0, 1, pure_test.shape)
noisy_input = pure + 0.35 * noise
noisy_input_test = pure_test + 0.35 * noise_test



#denoised_images = model.predict(noisy_input_test)

# Odszumiane
number_of_visualizations = 5
samples = noisy_input_test[:number_of_visualizations]
targets = test_labels[:number_of_visualizations]
denoised_images = model.predict(samples)

# Rysowanie
for i in range(0, number_of_visualizations):
  # Próbki i rekonstrukcja
  noisy_image = noisy_input_test[i][:, :, 0]
  pure_image  = pure_test[i][:, :, 0]
  denoised_image = denoised_images[i][:, :, 0]
  input_class = targets[i]

  fig, axes = plt.subplots(1, 3)
  fig.set_size_inches(8, 3.5)
  axes[0].imshow(noisy_image)
  axes[0].set_title('Noisy image')
  axes[1].imshow(pure_image)
  axes[1].set_title('Pure image')
  axes[2].imshow(denoised_image)
  axes[2].set_title('Denoised image')
  fig.suptitle(f'MNIST target = {input_class}')
  plt.show()

