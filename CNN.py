import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Activation, Dropout, MaxPooling2D
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import expand_dims

tf.device('/device:GPU:0')
model = Sequential()

model.add(Conv2D(32, (3,3), padding = 'same', input_shape=(80, 60,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (2,2), padding= 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (2,2), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])

model.summary()
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
		rescale=1./255,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
		'Images/Training',
		target_size=(80, 60),
		batch_size=32,
		class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
		'Images/Testing',
		target_size=(80, 60),
		batch_size=32,
		class_mode='binary')

history = model.fit_generator(
		train_generator,
		steps_per_epoch=2000,
		epochs=15,
		validation_data=validation_generator,
		validation_steps=800)
# Accuracy 65.82%
model = Model(inputs=model.inputs, outputs=model.layers[6].output)
img = load_img('Images/Training/Drowsy/1005.jpg', target_size=(80, 60))
img = img_to_array(img)
img = expand_dims(img, axis=0)
img = preprocess_input(img)
feature_maps = model.predict(img)
square = 8
ix = 1
for _ in range(square):
	for _ in range(square):
		ax = plt.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
		ix += 1
plt.show()