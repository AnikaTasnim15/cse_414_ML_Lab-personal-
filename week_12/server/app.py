
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import Input
import matplotlib.pyplot as plt

import os
from keras.applications.resnet50 import ResNet50
import tensorflow as tf

import json
import cv2
import json
import gradio as gr

#Diseable cuda
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

feature_model = ResNet50(include_top=True)

global inv_dict, word_count , captions_dict, extractor, model

extractor = Model(inputs = feature_model.input,outputs = feature_model.layers[-2].output)
model = tf.keras.models.load_model('./assets/FunctionalModel.hdf5')



with open('./assets/inv_dict.json') as f:
    inv_dict = json.load(f)


with open('./assets/word_count.json') as f:
    word_count = json.load(f)
    
with open("./assets/captions_dict.json") as f:
    captions_dict = json.loads(f.read())


def getImg(img):
    test_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    test_img = cv2.resize(img, (224,224))
    test_img = np.reshape(test_img, (1,224,224,3))    
    return test_img


def Gen_Text(img):

    MAX_LEN=39
    test_feature = extractor.predict(getImg(img)).reshape(1,2048)
    text_inp = ['startofseq']
    count = 0
    caption = ''
    while count < 25:
        count += 1

        encoded = []
        for i in text_inp:
            encoded.append(word_count[i])

        encoded = [encoded]

        encoded = pad_sequences(encoded, padding='post', truncating='post', maxlen=MAX_LEN)


        prediction = np.argmax(model.predict([test_feature, encoded]))

        sampled_word = inv_dict[str(prediction)]

        
            
        if sampled_word == 'endofseq':
            break
        else:
            caption = caption + ' ' + sampled_word

        text_inp.append(sampled_word)

    return img, caption[1:]



if __name__ == '__main__':
    iface = gr.Interface(fn = Gen_Text, 
                        inputs=gr.inputs.Image(), 
                        outputs=["image","text"])
    iface.launch(share=True)
    














