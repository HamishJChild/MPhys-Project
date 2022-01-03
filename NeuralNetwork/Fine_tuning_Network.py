''' This code takes model weights from a model pre-trained on Abraham galaxy images, and then re-trains the images on Efigi images for fine tuning'''


from __future__ import print_function
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from keras.models import Model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, LeakyReLU
from keras import optimizers
from keras import initializers
from keras import regularizers
from keras.optimizers import SGD, Adadelta
from keras.callbacks import CSVLogger, EarlyStopping
from sklearn.metrics import roc_curve
from sklearn.metrics import auc, f1_score
#initial parameters
epochs=5
batch_size= 32
target_size = (96,96)
num_classes=3
xcol='Image_path_2'
ycol1='binaryttype'
ycol=['Lenticular','Elliptical','Spiral']
class_names=['Lenticular','Elliptical','Spiral']
labels_path='../Data/EFIGI_ttype_labels.csv'
image_path='../Data/Efigi_images'
# load .csv file (input own file path here)
labels_all = pd.read_csv(labels_path, delimiter= ',')
print(labels_all.shape) #print shape
print('loaded the cvs file')

# change the file:// in the pathname on the matched table with a blank space so python can understand it
labels_all['Image_path_2'] = [q.replace("file://","") for q in labels_all['path']]

file_exists = [os.path.isfile(q) for q in labels_all['Image_path_2']]

labels = labels_all[file_exists]

# set up the data augmentations, not all are active
dftrain = ImageDataGenerator(zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
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
df=ImageDataGenerator(rescale=1./255.)




#Map the labels onto the images for training, validating and testing
train=df.flow_from_dataframe(
    dataframe = labels[:1000],
    directory= None,
    batch_size= 32,
    x_col = xcol, # column name in cvs file where the paths to the images are
    y_col= ycol,
    class_mode ='raw',
    target_size=target_size,
    validate_filenames=True,
    shuffle=False
    )

validate=df.flow_from_dataframe(
    dataframe=labels[500:3700],
    directory= None,
    batch_size= 32,
    x_col = xcol, # column name in cvs file where the paths to the images are
    y_col= ycol,
    class_mode ='raw',
    target_size=target_size,
    validate_filenames=True,
    shuffle=False
    )

test=df.flow_from_dataframe(
    dataframe =labels[3000:4458],
    directory= None,
    batch_size= 32,
    x_col = xcol, # column name in cvs file where the paths to the images are
    y_col= ycol,
    class_mode ='raw',
    target_size=target_size,
    validate_filenames=True,
    shuffle=False
    )




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
model.add(Dense(1000, activation= 'relu', name= 'dense_1'))

#model.add(Dropout(0.25, name='dropout4'))

model.add(Dense(3, activation= 'softmax', name='dense_3'))


print(model.summary())

#freeze training on feature detection layers
for layer in model.layers[:-9]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in model.layers:
    print(layer, layer.trainable)

# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer= 'adamax',
              metrics=['accuracy'])

# set up the step size
STEP_SIZE_TRAIN=train.n//train.batch_size
STEP_SIZE_VALID=validate.n//validate.batch_size
STEP_SIZE_TEST=test.n//test.batch_size +1

print(STEP_SIZE_TEST)


model.load_weights('../Abraham_models/abraham_pre_trained_weights.h5', by_name=True)
print('WEIGHTS LOADED')



# train the network 
model.fit_generator(generator=train,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=validate,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=epochs
)


# calculate the score by evaluating on the test dataset
score= model.evaluate_generator(generator=test,
steps=STEP_SIZE_TEST, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Make the predictions
pred_labels=model.predict_generator(test, steps= STEP_SIZE_TEST)


#pred_labels=np.concatenate(1(-pred_labels, pred_labels), axis=1)
df = pd.DataFrame(pred_labels)




test_data=labels.loc[3000:4458]
true_labels=test_data.loc(axis=1)['Lenticular','Elliptical','Spiral']
true_labels=true_labels.to_numpy()
df2=pd.DataFrame(true_labels)
df3=df.append(df2)
df3.to_csv('../Results/fine_tuned_predictions_finetuned.csv')



fpr_keras = dict()
tpr_keras=dict()
thresholds_keras=dict()
auc_keras=dict()


for i in range(num_classes):
  fpr_keras[i], tpr_keras[i], thresholds_keras[i] = roc_curve(true_labels[:,i], pred_labels[:,i])
  auc_keras[i] = auc(fpr_keras[i], tpr_keras[i])


# Plot the Reciever Operating Curve
plt.figure()
for i in range(num_classes):
        plt.plot(fpr_keras[i], tpr_keras[i], label= class_names[i] +  '(area = %0.2f)' % auc_keras[i])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Curve')
plt.legend(loc='best')
plt.show()
