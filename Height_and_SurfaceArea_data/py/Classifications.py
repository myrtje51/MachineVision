# evaluate logistic regression on encoded input
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
import pandas as pd
import cv2
import numpy as np
import os
from sklearn.svm import SVC

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

# load the model from file
encoder = load_model('encoder_2.0.h5')
#encoder = load_model('classification_encoder.h5')
# encode the train data
X_train_encode = encoder.predict(all_photos_train)
# encode the test data
X_test_encode = encoder.predict(all_photos_test)
# define the model
model = LogisticRegression()
# fit the model on the training set
model.fit(X_train_encode, train_labels)
# make predictions on the test set
yhat = model.predict(X_test_encode)
# calculate classification accuracy
acc = accuracy_score(test_labels, yhat)
print("Logistic Regression with encoder features:", acc)

# Building the SVM model
svmclf = SVC()
svmclf.fit(X_train_encode, train_labels)
yhatsvc = svmclf.predict(X_test_encode)
accsvc = accuracy_score(test_labels, yhatsvc)
# printing the accuracy of the non-linear model
print("SVM with encoder features:", accsvc)