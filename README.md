from tensorflow.keras.models import load_model
import flickrapi
import urllib
import tensorflow as tf
# Import the Sequential model and layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import optimizers
from tensorflow.keras import optimizers

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(300, 300, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
sgd = optimizers.SGD(lr = 0.001, decay = 1e-6, momentum = 0.9, nesterov = True)
#rmsprop
model.compile(loss = 'binary_crossentropy',
              optimizer = 'rmsprop',
             metrics=["accuracy",tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

batch_size = 10

# Training Augmentation configuration
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, 
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Testing Augmentation - Only Rescaling
test_datagen = ImageDataGenerator(rescale = 1./255)
#rescale=1./255
# Generates batches of Augmented Image data
train_generator = train_datagen.flow_from_directory('train 2/', target_size = (300, 300), 
                                                    batch_size = batch_size,
                                                    class_mode = 'binary') 

# Generator for validation data
validation_generator = test_datagen.flow_from_directory('test 2/', 
                                                        target_size = (300, 300),
                                                        batch_size = batch_size,
                                                        class_mode = 'binary')

model.fit_generator(train_generator,
                    epochs = 5 ,
                    validation_data = validation_generator,
                    verbose = 1)

# Evaluating model performance on Testing data
print("\n---------------------------\n")

loss, accuracy, precision, recall  = model.evaluate(validation_generator)
F1 = 2 * (precision * recall) / (precision + recall)
print("F1= ",F1)
print("\n---------------------------\n")


def modelx(new_acc):
 loss, accuracy, precision, recall = model.evaluate(validation_generator)
 if new_acc > accuracy :
  return True




import os.path
if os.path.isfile('Robot2.h5') is False :
     print('\nNew Accuracy: ', accuracy, '\nLoss: ', loss)

     model.save('Robot2.h5')
     model.save_weights('my_model_weights2.h5')

elif modelx(accuracy) is True :
    model.save('Robot2.h5')
    model.save_weights('my_model_weights2.h5')
