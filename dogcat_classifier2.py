from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras import optimizers
from keras.layers import Dense,Activation,Dropout
import os
import matplotlib.pyplot as plt
from PIL import Image
#from tensorflow.keras import datasets,layers,models
#from livelossplot import PlotLossesKera

base_path = os.getcwd()
#training_data_dir = os.path.join(base_path,"data","training")
test_data_dir = os.path.join(base_path,"test1")

training_data_dir = os.path.join(base_path,"train1")
#print(training_data_dir)
validation_data_dir = os.path.join(base_path,"validation1")
print(validation_data_dir)
#test_data_dir = r"./data/testing/"
IMAGE_WIDTH=64
IMAGE_HEIGHT=64
BATCH_SIZE=10
training_data_generator=ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)
validation_data_generator=ImageDataGenerator(rescale=1./255)
test_data_generator=ImageDataGenerator(rescale=1./255)
#print(training_data_generator.flow_from_directory(training_data_dir))

#datagen=ImageDataGenerator()
training_generator=training_data_generator.flow_from_directory(
    training_data_dir,
    target_size=(IMAGE_WIDTH,IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="binary")
validation_generator=validation_data_generator.flow_from_directory(
    validation_data_dir,
    target_size=(IMAGE_WIDTH,IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="binary")
target_generator=test_data_generator.flow_from_directory(
    test_data_dir,
    target_size=(IMAGE_WIDTH,IMAGE_HEIGHT),
    batch_size=1,
    class_mode="binary",
    shuffle=False)


model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(64,64,3), activation='relu'))
model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))
    
model.compile(loss='binary_crossentropy',
            optimizer=optimizers.RMSprop(lr=0.0001),
            metrics=['accuracy'])


history=model.fit_generator(
    training_generator,
    steps_per_epoch=len(training_generator.filenames),
    epochs=4,
    validation_data=validation_generator,
    validation_steps=len(validation_generator.filenames))



print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

import numpy as np
from keras.preprocessing import image
img=Image.open('12.jpg')
plt.imshow(img)
test_image=image.load_img('12.jpg',target_size=(64,64))
test_image=image.img_to_array(test_image)
#test_image=test_image.reshape((28,28))
#plt.imshow(test_image)

test_image=np.expand_dims(test_image,axis=0)
result=model.predict(test_image)
if result[0][0]>=0.5:
    prediction="dog"

else:
    prediction='cat'
print(prediction)
plt.title(prediction)
plt.show()

                











