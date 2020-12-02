import numpy as np
import pandas as pd
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import string
import random

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


from keras.preprocessing import sequence, text
from keras.utils import np_utils
import keras
#!pip install tqdm
from tqdm import tqdm
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D

# process data
df = pd.read_csv('data/combined_tags_final.csv')
df = df.drop_duplicates()
df = df.dropna()
df.isnull().sum()
df = df.reset_index(drop = True)
len(df)

def load_data(data):
    x = data['tweet'].tolist()
    y = data['hashtags'].tolist()
    return x, y

X, y = load_data(df)
print('size:', len(X))


stop_words = set(stopwords.words('english')) 
word_sets = []

for i in range(len(X)):
    word_tokens = word_tokenize(X[i]) 
    word_sets.append(word_tokens)

lemmatizer = nltk.WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')
transtbl = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

def preprocessing(line: str) -> str:
    line = line.replace('<br />', '').translate(transtbl)
    
    tokens =[lemmatizer.lemmatize(t.lower(),'v') 
             for t in nltk.word_tokenize(line) 
             if t.lower() not in stopwords]
    
    return ' '.join(tokens)

X = list(map(preprocessing, X)) # X is tweets_to.list()
y_du = pd.get_dummies(df['hashtags']).values
X_train, X_test, y_train, y_test = train_test_split(X, y_du, test_size=0.2, random_state=42)

#keras tokenizer
token = text.Tokenizer(num_words=None)
max_len = 10

token.fit_on_texts(list(X_train) + list(X_test))
xtrain_seq = token.texts_to_sequences(X_train)
xvalid_seq = token.texts_to_sequences(X_test)

xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)
word_index = token.word_index

ytrain_enc = np_utils.to_categorical(y_train)
yvalid_enc = np_utils.to_categorical(y_test)

embedding_matrix = np.zeros((len(word_index) + 1, 100))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


model = Sequential()
model.add(Embedding(len(word_index) + 1,
                     100,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
model.add(SpatialDropout1D(0.3))
model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.8))

model.add(Dense(3154))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(xtrain_pad, y=ytrain_enc, batch_size=128, epochs=100, verbose=1, validation_data=(xvalid_pad, yvalid_enc))
pred = model.predict(xvalid_pad)
res = pd.DataFrame(pred)
res.idxmax(axis="columns").value_counts()
# 1468 joebiden 79019
# 8 2020election 553
# 1145 giveaway 182
# 624 coronavirus 66
# 1023 foia 51
# 398 breaking 31
# 810 donaldtrump 1
dumm = pd.get_dummies(df['hashtags'])
dumm.iloc[:,810:811]
pre = res.idxmax(axis="columns")
pre = pd.DataFrame(pre)
temp = df.copy()
temp['lstm_pred'] = None
joebiden_tags = list(pre[pre['tag']==1468].index)
election_tags = list(pre[pre['tag']==8].index)
giveaway_tags = list(pre[pre['tag']==1145].index)
coronavirus_tags = list(pre[pre['tag']==624].index)
foia_tags = list(pre[pre['tag']==1023].index)
breaking_tags = list(pre[pre['tag']==398].index)
donaldtrump_tags = list(pre[pre['tag']==810].index)
for j in joebiden_tags:
    temp['lstm_pred'][j] = 'joebiden'
for j in election_tags:
    temp['lstm_pred'][j] = '2020election'
for j in giveaway_tags:
    temp['lstm_pred'][j] = 'giveaway'
for j in coronavirus_tags:
    temp['lstm_pred'][j] = 'coronavirus'
for j in foia_tags:
    temp['lstm_pred'][j] = 'foia'
for j in breaking_tags:
    temp['lstm_pred'][j] = 'breaking'
for j in donaldtrump_tags:
    temp['lstm_pred'][j] = 'donaldtrump'
lstm_pred = temp['lstm_pred'][0:79903,]
a, b, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(classification_report(y_test,lstm_pred,labels = ['joebiden','donaldtrump','2020election','vote','giveaway','coronavirus','foia','breaking']))


