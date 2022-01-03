'''The following code gets the Abraham images of galaxies and uses them to train and test a neural network. The weights of the NN are then saved.
A new model is then set up and the weights are loaded in order to demonstrate how each layer 'views' an image being passed through the network.'''
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import glob
import imageio as im
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import models
from keras.models import Model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, LeakyReLU
from keras import optimizers
from keras import initializers
from keras import regularizers
from tensorflow.keras.optimizers import SGD, Adadelta
from keras.callbacks import CSVLogger, EarlyStopping
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from keras.preprocessing import image

epochs=15
batch_size=64
target_size = (96,96)
num_classes=3
class_names=['Lenticular','Elliptical','Spiral',]
xcol='path'
ycol1='binaryttype'
ycol=['Lenticular','Elliptical','Spiral']
labels_path='../Abraham_ttype_labels.csv'

labels_all=pd.read_csv(labels_path, delimiter=',')

print(labels_all.shape) #print shape
print('loaded the cvs file')



labels = labels_all.sample(frac=1).reset_index(drop=True) #shuffling the data
print('Shuffled the data')


# set up the data augmentations, not all are active
dftrain = ImageDataGenerator(
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        #width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        #height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=1./255.)
df=ImageDataGenerator(rescale=1./255.
    )

# The data is split into train, validate and testing sets.

train=dftrain.flow_from_dataframe(
    dataframe=labels[:10956],
    directory= None,
    batch_size= 32,
    x_col = xcol, # column name in cvs file where the paths to the images are
    y_col= ycol,
    class_mode ='raw',
    target_size=target_size,
    validate_filenames=True
    )

validate =df.flow_from_dataframe(
    dataframe=labels[10957:12326],
    directory= None,
    batch_size= 32,
    x_col = xcol, # column name in cvs file where the paths to the images are
    y_col= ycol,
    class_mode ='raw',
    target_size=target_size,
    validate_filenames=True
    )

test=df.flow_from_dataframe(
    dataframe=labels[12327:13696],
    directory= None,
    batch_size= 32,
    x_col = xcol, # column name in cvs file where the paths to the images are
    y_col= ycol,
    class_mode ='raw',
    target_size=target_size,
    validate_filenames=True
    )

# Here the Neural network is set up

input_shape=(96,96,3)

#set up the network
model = Sequential()
model.add(Conv2D(32, (6,6), padding='same',kernel_initializer= 'orthogonal' ,input_shape= input_shape,name='Conv1'))
model.add(LeakyReLU(alpha= 0.1))

model.add(Conv2D(64, (5,5), padding='same', name='Conv2'))
model.add(LeakyReLU(alpha= 0.1))
model.add(MaxPooling2D(pool_size=(2,2), name='MPool1'))

model.add(Conv2D(128, (2,2), padding='same', name='Conv3'))
model.add(LeakyReLU(alpha= 0.1))
model.add(MaxPooling2D(pool_size=(2,2), name='MPool2'))

#model.add(Dropout(0.25, name='dropout2'))

model.add(Conv2D(128, (3,3), padding = 'same', name='Conv4'))
model.add(LeakyReLU(alpha= 0.1))
#model.add(Dropout(0.25, name='dropout3'))


model.add(Flatten( name='flatten'))
model.add(Dense(1000, activation= 'relu', name= 'dense1'))

#model.add(Dropout(0.25, name='dropout4'))

model.add(Dense(3, activation= 'softmax', name='dense3'))

print(model.summary())

# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adamax',
              metrics=['accuracy'])

# set up the step size
STEP_SIZE_TRAIN=train.n//train.batch_size   
STEP_SIZE_VALID=validate.n//validate.batch_size 
STEP_SIZE_TEST=test.n//test.batch_size +1   

print(STEP_SIZE_TEST)

csv_logger=CSVLogger('../Abraham_models/abraham_epoch_values.csv')
earlystopping=EarlyStopping(monitor='acc', mode='max', patience=10)

# train the network 
model.fit_generator(generator=train,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=validate,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=epochs,
                    callbacks=[csv_logger, earlystopping]
)

# calculate the score by evaluating on the test dataset
score= model.evaluate_generator(generator=test,
steps=STEP_SIZE_TEST, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the weights and model to a .h5 file
print("Saving model...")
model.save_weights('../Abraham_models/abraham_pre_trained_weights.h5',overwrite=True)
model.save('../Abraham_models/abraham_pre_trained_network.h5') 
print('Saved model and weights')

pred_labels=model.predict_generator(test, steps= STEP_SIZE_TEST)
print(pred_labels.shape)

df = pd.DataFrame(pred_labels)

df.to_csv('../efigi_predictions.csv')

test_data=labels.loc[3730:4458]
true_labels=test_data.loc(axis=1)['Lenticular','Elliptical','Spiral']
print(true_labels)
true_labels=true_labels.to_numpy()

#y_test_classes = np.zeros_like(true_labels)

print(true_labels)
print(pred_labels)

fpr_keras = dict()
tpr_keras=dict()
thresholds_keras=dict()
auc_keras=dict()


for i in range(num_classes):
  fpr_keras[i], tpr_keras[i], thresholds_keras[i] = roc_curve(true_labels[:,i], pred_labels[:,i])
  auc_keras[i] = auc(fpr_keras[i], tpr_keras[i])



for i in range(num_classes):
    plt.figure()
    plt.plot(fpr_keras[i], tpr_keras[i], label='ROC curve (area = %0.2f)' % auc_keras[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Curve')
    plt.legend(loc='best')
    plt.show()

# This section sets up the model and loads the weights again. The images produced by each individual layer for a figure being passed through the model are shown


input_shape=(96,96,3)
model = Sequential()
model.add(Conv2D(32, (6,6), padding='same',kernel_initializer= 'orthogonal' ,input_shape= input_shape,name='Conv1'))
model.add(LeakyReLU(alpha= 0.1, name='leaky1'))

model.add(Conv2D(64, (5,5), padding='same', name='Conv2'))
model.add(LeakyReLU(alpha= 0.1, name='leaky2'))
model.add(MaxPooling2D(pool_size=(2,2), name='MPool1'))

model.add(Conv2D(128, (2,2), padding='same', name='Conv3'))
model.add(LeakyReLU(alpha= 0.1,name='leaky3'))
model.add(MaxPooling2D(pool_size=(2,2), name='MPool2'))

#model.add(Dropout(0.25, name='dropout2'))

model.add(Conv2D(128, (3,3), padding = 'same', name='Conv4'))
model.add(LeakyReLU(alpha= 0.1, name='leaky4'))
#model.add(Dropout(0.25, name='dropout3'))


model.add(Flatten( name='flatten'))
model.add(Dense(1000, activation= 'relu', name= 'dense_1'))

#model.add(Dropout(0.25, name='dropout4'))

model.add(Dense(3, activation= 'softmax', name='dense_3'))



print(model.summary())

# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adamax',
              metrics=['accuracy'])

# model.load_weights('../CIRAF10_Models/fashion_mnist_pre_trained_weights.h5', by_name=True)
model.load_weights('../Abraham_models/abraham_pre_trained_weights.h5', by_name=True)

img_path = '../10249.0.jpg'

img = image.load_img(img_path, target_size=(96,96))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

plt.imshow(img_tensor[0])
plt.show()

layer_outputs = [layer.output for layer in model.layers[:12]] # Extracts the outputs of the top 12 layers
activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
activations = activation_model.predict(img_tensor) # Returns a list of five Numpy arrays: one array per layer activation


layer_names = []
for layer in model.layers[:12]:
    layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] # Number of features in the feature map
    size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
    n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row):
            channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
            channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, # Displays the grid
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
