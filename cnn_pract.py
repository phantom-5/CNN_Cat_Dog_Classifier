#CNN for Dog - Cat Classification

#Step -1 Bulding the CNN
#Imports
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

#Step 2- Initialising the CNN
classifier=Sequential()

#Step 3- Convolution Layer
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
# 32 - no. of filters to use, (3,3) filter matrix row-col
# input_shape - 3 means 3 channels(Color channels) 64X64 size, increase size for better accuracy

#Step 4- Pooling
classifier.add(MaxPooling2D(pool_size=[2,2]))
#generally a 2X2 matrix is taken , as it allows no loss of information

# Adding a second convolution layer and pooling layer
classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=[2,2]))

#Step -5 Flattening ( Creating the input vector)
classifier.add(Flatten())

#Step -6 Full Connection ( Adding the first hidden layer and the output layer)
classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))

#Adding Dropout
classifier.add(Dropout(p=0.1))

#Step 7- Compiling the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Step 8- Fitting the CNN to images
from keras.preprocessing.image import ImageDataGenerator
#ImageDataGenerator basically does augmentation of images to create new images
# with various transfomrations so as to increase the sample size
#rescale does scaling of pixel values
train_datagen = ImageDataGenerator(
        rescale=1./255,   #pixels have values between 0 and 1
        shear_range=0.2,  
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64,64),   #should be same as in step 3
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)

# Single Prediction
import numpy as np
from keras.preprocessing import image
test_image=image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size=(64,64))
test_image=image.img_to_array(test_image) #converting to 3D array
test_image=np.expand_dims(test_image,axis=0) #converting to 4D for predict method to work
result=classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction='dog'
else:
    prediction='cat'