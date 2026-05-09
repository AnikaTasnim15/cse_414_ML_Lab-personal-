from keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import Xception
import numpy as np
import pandas as pd
import pickle
import cv2
import os
from glob import glob
from pickle import dump, load
from keras.models import Model
import json

feature_model = Xception(include_top=True)

extractor = Model(inputs = feature_model.input,outputs = feature_model.layers[-2].output)
print(extractor.summary())



images_path = '../Data/BNG_9k/images/'
images = glob(images_path+'*.png')
print(len(images))



images_features = {}
count = 0
for i in images:
    img = cv2.imread(i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (299,299))

    
    
    img = img.reshape(1,299,299,3)
    pred = extractor.predict(img).reshape(2048,)
        
    img_name = i.split('/')[-1]
    
    images_features[img_name] = pred
    
    count += 1
        
    if count % 100 == 0:
        print(count)

        
with open('../Logs/Xception_BNG9k/image_features.pickle', "wb" ) as pickle_f:
    pickle.dump(images_features, pickle_f)


print('Success', len(images_features))


# img = cv2.imread('../Data/BNG_9k/images/1000.png')
# print(img.shape)
# print(img)
# cv2.imshow('img', img)
# cv2.waitKey(0)

# for i in images:
#     print(i.split('/')[-1])
#     img = cv2.imread(i)
#     cv2.imshow('img', img)
#     cv2.waitKey(0)

