
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
from sklearn.linear_model import LogisticRegression

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report


# process data
df = pd.read_csv('data/combined_tags_final.csv')
df = df.drop_duplicates()
df = df.dropna()
df.isnull().sum()
df = df.reset_index(drop = True)
len(df)


# Baseline Model
tags = set()
for i in range(len(df)):
     tags.add(df['hashtags'][i])
        
d_test = df.copy()
d_test = d_test.drop(['hashtags'],axis= 1)
d_test['test_tags'] = None

tags = list(tags)
for j in range(len(df)):
    sel = random.randint(0,3153)
    d_test['test_tags'][j] = tags[sel]

y_true = np.array(df['hashtags'])
y_pred = np.array(d_test['test_tags'])

print(classification_report(y_true,y_pred, labels = ['joebiden', 'donaldtrump', 'vote','2020election','usa','kamalaharris']))


# Naive Bayes model
s = df['hashtags'].value_counts()
res = s.to_dict()

hash_4 = []
for key in res:
    if res[key] < 20:
        hash_4.append(key)
        
for i in range(len(hash_4)):
    df.drop(df.index[df['hashtags'] == hash_4[i]], inplace = True)


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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Vocabulary
all_words = [w for line in X_train for w in line.split()]
voca = nltk.FreqDist(all_words)

def train_with_n_topwords(n: int) -> tuple:
    topwords = [fpair[0] for fpair in list(voca.most_common(n))]
    
    # Vectorize with N-Gram
    vec = TfidfVectorizer(vocabulary=topwords)
    
    # Feature Extraction
    train_features = vec.fit_transform(X_train)
    test_features  = vec.transform(X_test)
    
    # Train Model
    model_mnb = MultinomialNB()
    model_mnb.fit(train_features, y_train)
    
    # Predict
    pred = model_mnb.predict(test_features)
   
    return metrics.accuracy_score(y_test,pred), pred,model_mnb





# Select Top N Words
possible_n = [500 * i for i in range(1, 30)]

tfidf_accuracies = []
pred = []


for i, n in enumerate(possible_n):
    tfidf_accuracies.append(train_with_n_topwords(n)[0])
    pred.append(train_with_n_topwords(n)[1])

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(possible_n, tfidf_accuracies, label='Tf-idf')
plt.xlabel('Number of topwords')
plt.ylabel('Accuracy')
plt.grid()
plt.legend()
plt.show()

tfidf_accuracies[max(tfidf_accuracies)] # when top words is 9000 accuracy is the best
topwords = [fpair[0] for fpair in list(voca.most_common(9000))]
    
# Vectorize with N-Gram
vec = TfidfVectorizer(vocabulary=topwords)
    
# Feature Extraction
train_features = vec.fit_transform(X_train)
test_features  = vec.transform(X_test)

model_mnb = MultinomialNB()
model_mnb.fit(train_features,y_train)
    
# Predict
pred = model_mnb.predict(test_features)
print(classification_report(y_test, pred, labels = ['joebiden', 'donaldtrump', 'vote','2020election','usa','kamalaharris']))
nb_count = MultinomialNB()
topwords = [fpair[0] for fpair in list(voca.most_common(9000))]
    
# Vectorize with N-Gram
cou= CountVectorizer(vocabulary=topwords)
    
# Feature Extraction
train_count = cou.fit_transform(X_train)
test_count  = cou.transform(X_test)

nb_count.fit(train_count, y_train)
# Predict
pred_count = nb_count.predict(test_count)
print(classification_report(y_test, pred_count, labels = ['joebiden', 'donaldtrump', 'vote','2020election','usa','kamalaharris']))


# Xgboost
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# train_features = vec.fit_transform(X_train)
# test_features  = vec.transform(X_test)

# def multiclass_logloss(actual, predicted, eps=1e-15):
#     # Convert 'actual' to a binary array if it's not already:
#     if len(actual.shape) == 1:
#         actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
#         for i, val in enumerate(actual):
#             actual2[i, val] = 1
#         actual = actual2

#     clip = np.clip(predicted, eps, 1 - eps)
#     rows = actual.shape[0]
#     vsota = np.sum(actual * np.log(clip))
#     return -1.0 / rows * vsota

# clf = xgb.XGBClassifier(max_depth=7, n_estimators=2, colsample_bytree=0.8, 
#                         subsample=0.8, nthread=10, learning_rate=0.1)
# clf.fit(train_features, y_train)
# predictions = clf.predict_proba(test_features)

# print ("logloss: %0.3f " % multiclass_logloss(y_test, predictions))


# lightgbm
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# train_features = vec.fit_transform(X_train)
# test_features  = vec.transform(X_test)

# # Light GBM
# import lightgbm as lgb
# train_data = lgb.Dataset(train_features, label=y_train)
# test_data = lgb.Dataset(test_features, label=y_test)
# #setting parameters for lightgbm
# param ={'num_leaves':90, 'objective':'regression_l2', 'num_leaves':10,'max_depth':5,'learning_rate':0.3,'max_bin':400,'boosting':'dart'}
# param['metric'] = ['auc', 'l2']

# lightgbm_model = lgb.train(param, train_data,valid_sets = test_data, num_boost_round = 50, early_stopping_rounds = 100)


# Random Forest with Tf-idf
classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 42,max_features = 'sqrt',bootstrap=True,max_depth = 100)
classifier.fit(train_features, y_train)
pred_rf = classifier.predict(test_features)
print(classification_report(y_test, pred_rf, labels = ['joebiden', 'donaldtrump', 'vote','2020election','usa','kamalaharris']))

rf = RandomForestClassifier(random_state = 42,bootstrap = True)
param_grid = { 
    'n_estimators': [20,50,100],
    'max_features': ['sqrt', 'log2'],
    'max_depth' : [5,7],
    'criterion' :['gini', 'entropy']
    }

cv_rf = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
cv_rf.fit(train_features, y_train)

# Random Forest with Word Count
classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 42,max_features = 'sqrt',bootstrap=True,max_depth = 100)
classifier.fit(train_count, y_train)
pred_co_rf = classifier.predict(test_count)
print(classification_report(y_test, pred_co_rf, labels = ['joebiden', 'donaldtrump', 'vote','2020election','usa','kamalaharris']))

rf = RandomForestClassifier(random_state = 42,bootstrap = True)
param_grid = { 
    'n_estimators': [20,50,100],
    'max_features': ['sqrt', 'log2'],
    'max_depth' : [5,7],
    'criterion' :['gini', 'entropy']
    }

cv_rf = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
cv_rf.fit(train_count, y_train)







