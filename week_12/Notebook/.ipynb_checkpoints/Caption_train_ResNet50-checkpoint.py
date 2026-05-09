import pickle 
import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
import matplotlib.pyplot as plt

def preprocessed(txt):
    modified = txt
    modified = 'startofseq ' + modified + ' endofseq'
    return modified






file = open('../Logs/ResNet50_BNG9k/image_features.pickle', "rb")
images_features = pickle.load(file)
file.close()

caption_path = '../Data/BNG_9k/Captions.txt'
captions = open(caption_path, 'rb').read().decode('utf-8').split('\n')

captions_dict = {}
for i in captions:
    try:
        img_name = i.split('\t')[0][:-2] 
        caption = i.split('\t')[1]
        if 'images\\'+img_name in images_features:
            if img_name not in captions_dict:
                captions_dict[img_name] = [caption]
                
            else:
                captions_dict[img_name].append(caption)
            
    except:
        pass
    

    
for img_name, texts in captions_dict.items():
    for text_content in texts:
        captions_dict[img_name][texts.index(text_content)] = preprocessed(text_content)
        
        

word_count = {}
count = 1
for img_name, texts in captions_dict.items():
    for text_content in texts:
        for word in text_content.split():
            if word not in word_count:
                word_count[word] = count
                count += 1


for img_name, texts in captions_dict.items():
    for text_content in texts:
        encoded = []
        for word in text_content.split():  
            encoded.append(word_count[word])
        captions_dict[img_name][texts.index(text_content)] = encoded
        
        
    
MAX_LEN = 0
for img_name, texts in captions_dict.items():
    for text_content in texts:
        if len(text_content) > MAX_LEN:
            MAX_LEN = len(text_content)
            
global VOCAB_SIZE
VOCAB_SIZE = len(word_count)         

def generator(photo, caption):
    n_samples = 0
    
    X = []
    y_in = []
    y_out = []
    
    for k, vv in caption.items():
        for v in vv:
            for i in range(1, len(v)):
                X.append(photo['images\\'+k])

                in_seq= [v[:i]]
                out_seq = v[i]

                in_seq = pad_sequences(in_seq, maxlen=MAX_LEN, padding='post', truncating='post')[0]
                out_seq = to_categorical([out_seq], num_classes=VOCAB_SIZE + 1)[0]

                y_in.append(in_seq)
                y_out.append(out_seq)
            
    return X, y_in, y_out
 


X, y_in, y_out = generator(images_features, captions_dict)
X = np.array(X)
y_in = np.array(y_in, dtype='float64')
y_out = np.array(y_out, dtype='float64')


embedding_size = 128
max_len = MAX_LEN
vocab_size = len(word_count) + 1


image_model = Sequential()
image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
image_model.add(RepeatVector(max_len))

print('Image model summary',image_model.summary(),sep='\n')


language_model = Sequential()

language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
language_model.add(LSTM(256, return_sequences=True))
language_model.add(TimeDistributed(Dense(embedding_size)))




print('Language model summary',language_model.summary(),sep='\n')


conca = Concatenate()([image_model.output, language_model.output])
x = LSTM(128, return_sequences=True)(conca)
x = LSTM(512, return_sequences=False)(x)
x = Dense(vocab_size)(x)
out = Activation('softmax')(x)


model = Model(inputs=[image_model.input, language_model.input], outputs = out)
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])


print('concatenated model summary',model.summary(),sep='\n')







history = model.fit([X, y_in], y_out, batch_size=512, epochs=50)



