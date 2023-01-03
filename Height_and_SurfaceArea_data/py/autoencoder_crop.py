import keras
from matplotlib import pyplot as plt
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, Flatten, Reshape, Input, InputLayer, LeakyReLU, BatchNormalization, Dropout 
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
import cv2
import numpy as np
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import layers
from keras_tuner import RandomSearch

train_data = 'Sativa_AS/training.gz'
test_data = 'Sativa_AS/test.gz'
conf_table = pd.read_excel('PileUp_dronedata_Feb2022.xlsx')
conf_table2 = conf_table.dropna(subset=['subgroup'])
subgroup = conf_table2[['Plot.ID','subgroup']]

# Read photos training set
all_photos_train = []
train_labels = []
sativa_ILs = os.listdir("training")
for m in sativa_ILs:
    IL = m[6:11]
    sg = subgroup.loc[subgroup['Plot.ID'] == IL]['subgroup'].values[0]
    train_labels.append(sg)
    img = cv2.imread("training/" + m)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (400, 500))
    clip = img[60:160, 40:140]
    all_photos_train.append(clip)

all_photos_train = np.stack(all_photos_train).astype('uint8')
train_labels = pd.factorize(train_labels)[0]
# Read photos test set
all_photos_test = []
test_labels = []
sativa_ILs = os.listdir("test")
for m in sativa_ILs:
    IL = m[6:11]
    sg = subgroup.loc[subgroup['Plot.ID'] == IL]['subgroup'].values[0]
    test_labels.append(sg)
    img = cv2.imread("test/" + m)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (400, 500))
    clip = img[60:160, 40:140]
    all_photos_test.append(clip)


#Data augmentation
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

for X_batch, y_batch in datagen.flow(all_photos_train, train_labels, batch_size=100, shuffle=False):
    break

all_photos_train = np.concatenate((all_photos_train, X_batch))
train_labels = np.concatenate((train_labels, y_batch))

all_photos_test = np.stack(all_photos_test).astype('uint8')
test_labels = pd.factorize(test_labels)[0]

# Transform RGB values from 0 to 255 to -0.5 to 0.5 because machine learning programs usually don't work with 
# high values in the 100s 
all_photos_train = all_photos_train.astype('float32') / 255.0
all_photos_test = all_photos_test.astype('float32') / 255.0

# Since the autoencoder by Keras has a hard time handling categorical data we apply one-hot encoding to our labels. 
train_Y_pd = pd.get_dummies(train_labels)
train_Y_one_hot = train_Y_pd.to_numpy()
test_Y_pd = pd.get_dummies(test_labels)
test_Y_one_hot = test_Y_pd.to_numpy()

train_X,valid_X,train_label,valid_label = train_test_split(all_photos_train,train_Y_one_hot,test_size=0.2,random_state=13)

def show_image(x):
    plt.imshow(np.clip(x + 0.5, 0, 1))

print("Check the range of RGB values: \n", all_photos_train.max(), all_photos_test.min())

def build_autoencoder(img_shape, code_size):
    # The encoder
    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    encoder.add(Flatten())
    encoder.add(BatchNormalization())
    encoder.add(Dense(code_size, activation=LeakyReLU(alpha=0.1)))

    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer((code_size,)))
    decoder.add(BatchNormalization())
    decoder.add(Dense(np.prod(img_shape), activation=LeakyReLU(alpha=0.1)))
    decoder.add(Reshape(img_shape))

    return encoder, decoder

def encoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(0.25)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    pool1 = Dropout(0.25)(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(0.25)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    pool2 = Dropout(0.25)(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(0.25)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
#    conv3 = Dropout(0.25)(conv3)
#    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)
#    conv4 = BatchNormalization()(conv4)
#    conv4 = Dropout(0.25)(conv4)
#    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
#    conv4 = BatchNormalization()(conv4)
    return conv3

def decoder(conv5):    
    #decoder
#    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4) #7 x 7 x 128
#    conv5 = BatchNormalization()(conv5)
#    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
#    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5) #7 x 7 x 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling2D((2,2))(conv6) #14 x 14 x 64
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 = UpSampling2D((2,2))(conv7) # 28 x 28 x 32
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded

def fc(enco):
    flat = Flatten()(enco)
    den = Dense(128, activation='relu')(flat)
    out = Dense(7, activation='softmax')(den)
    return out

def build_model(hp):          #hp means hyper parameters
    model=Sequential()
    model.add(Flatten(input_shape=(100,100,3)))
    #providing range for number of neurons in a hidden layer
    model.add(Dense(units=hp.Int('num_of_neurons',min_value=32,max_value=512,step=32),
                                    activation='relu'))
    #output layer
    model.add(Dense(10,activation='softmax'))
    #compiling the model
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',values=[1e-2, 1e-3, 1e-4])),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

# Build the encoder and decoder needed to fit the autoencoder on. 
IMG_SHAPE = all_photos_test.shape[1:]

inp = Input(IMG_SHAPE)

# Create the model with a loss of mse 
autoencoder = Model(inp, decoder(encoder(inp)))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())

print(autoencoder.summary())

encode = encoder(inp)
full_model = Model(inp,fc(encode))

for l1,l2 in zip(full_model.layers[:19],autoencoder.layers[0:19]):
    l1.set_weights(l2.get_weights())
    
#for layer in full_model.layers[0:19]:
#    layer.trainable = False
    
#full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])

#full_model.summary()

#classify_train = full_model.fit(train_X, train_label, batch_size=64,epochs=50,verbose=1,validation_data=(valid_X, valid_label))

#full_model.save_weights('autoencoder_classification_firsttest.h5')
#full_model.save('classification_encoder.h5')

for layer in full_model.layers[0:19]:
    layer.trainable = True

full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics=['accuracy'])
classify_train = full_model.fit(train_X, train_label, batch_size=64,epochs=25,verbose=1,validation_data=(valid_X, valid_label))
full_model.save_weights('classification_complete_firsttest.h5')
full_model.save('encoder_3.0.h5')

# saving the model so that we can analyse it in Jupyter Notebook for example
#autoencoder.save('autoencoder_model_test.h5')


accuracy = classify_train.history['accuracy']
val_accuracy = classify_train.history['val_accuracy']
loss = classify_train.history['loss']
val_loss = classify_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('model_accuracy_test.png')
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('model_loss_test.png')

# Predict using the autoencoder values. 

test_eval = full_model.evaluate(all_photos_test, test_Y_one_hot, verbose=0)

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

predicted_classes = full_model.predict(all_photos_test)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)

correct = np.where(predicted_classes==test_labels)[0]
print("Found %d correct labels" % len(correct))

incorrect = np.where(predicted_classes!=test_labels)[0]
print("Found %d incorrect labels" % len(incorrect))

target_names = ["Class {}".format(i) for i in range(7)]
print(classification_report(test_labels, predicted_classes, target_names=target_names))

# Predict using 

def visualize(img,encoder,decoder, number):
    """Draws original, encoded and decoded images"""
    # img[None] will have the same shape as the model input
    code = encoder.predict(img[None])[0]
    reco = decoder.predict(code[None])[0]

    plt.subplot(1,3,1)
    plt.title("Original")
    show_image(img)

    plt.subplot(1,3,2)
    plt.title("Code")
    plt.imshow(code.reshape([code.shape[-1]//2,-1]))

    plt.subplot(1,3,3)
    plt.title("Reconstructed")
    show_image(reco)
    name = str(number) + "_image_reconstruction4.png"
    plt.savefig(name)

extracted_features = [] 
    
    