#!/usr/bin/env python
# coding: utf-8

# In[126]:


import pandas as pd
import re
import numpy as np
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sqlite3 import Error
from sklearn.ensemble import RandomForestClassifier
import sqlite3
import pickle
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv(r'C:\Hackothon\Amozon\Dataset/train.csv')
test = pd.read_csv(r'C:\Hackothon\Amozon\Dataset/test.csv')
train.head(5)


# In[3]:


train = train.rename(columns = {"Review Text": "ReviewText", 
                                   "Review Title":"ReviewTitle" }) 
test = test.rename(columns = {"Review Text": "ReviewText", 
                                   "Review Title":"ReviewTitle" }) 


# In[127]:


def freq_words(x, terms = 30): 
  all_words = ' '.join([text for text in x]) 
  all_words = all_words.split() 
  fdist = nltk.FreqDist(all_words) 
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())}) 
  
  # selecting top 20 most frequent words 
  d = words_df.nlargest(columns="count", n = terms) 
  
  # visualize words and frequencies
  plt.figure(figsize=(12,15)) 
  ax = sns.barplot(data=d, x= "count", y = "word") 
  ax.set(ylabel = 'Word') 
  plt.show()
  
# print 100 most frequent words 
freq_words(train['ReviewText'], 100)
train['ReviewText'] = train['ReviewText'].apply(lambda x: remove_stopwords(x))


# In[4]:


stemmer = PorterStemmer()
words = stopwords.words("english")
train['ReviewText'] = train['ReviewText'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())


# In[5]:


vectorizer = TfidfVectorizer(min_df= 3, stop_words="english", sublinear_tf=True, norm='l2', ngram_range=(1, 2))
final_features = vectorizer.fit_transform(train['ReviewText']).toarray()
final_features.shape


# In[18]:


X_test.head(4)


# In[20]:


X_train.head(3)


# In[50]:


#first we split our dataset into testing and training set:
# this block is to split the dataset into training and testing set 
X = train['ReviewText']
Y = train['topic']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
# instead of doing these steps one at a time, we can use a pipeline to complete them all at once
pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=1200)),
                     ('clf', RandomForestClassifier())])
# fitting our model and save it in a pickle for later use
model = pipeline.fit(X_train, y_train)
with open('RandomForest.pickle', 'wb') as f:
    pickle.dump(model, f)
ytest = np.array(y_test)
# confusion matrix and classification report(precision, recall, F1-score)
print(classification_report(ytest, model.predict(X_test)))
print(confusion_matrix(ytest, model.predict(X_test)))


# In[85]:


import xgboost as xgb
from xgboost import XGBClassifier
#first we split our dataset into testing and training set:
# this block is to split the dataset into training and testing set 
X = train['ReviewText']
Y = train['topic']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
# instead of doing these steps one at a time, we can use a pipeline to complete them all at once
pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=1200)),
                     ('clf', XGBClassifier())])
# fitting our model and save it in a pickle for later use
model = pipeline.fit(X_train, y_train)
with open('XGBClassifier.pickle', 'wb') as f:
    pickle.dump(model, f)
ytest = np.array(y_test)
# confusion matrix and classification report(precision, recall, F1-score)
print(classification_report(ytest, model.predict(X_test)))
print(confusion_matrix(ytest, model.predict(X_test)))


# In[104]:


from sklearn.linear_model import LogisticRegression
#first we split our dataset into testing and training set:
# this block is to split the dataset into training and testing set 
X = train['ReviewText']
Y = train['topic']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
# instead of doing these steps one at a time, we can use a pipeline to complete them all at once
pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=1200)),
                     ('clf', LogisticRegression())])
# fitting our model and save it in a pickle for later use
model = pipeline.fit(X_train, y_train)
with open('LogisticRegression.pickle', 'wb') as f:
    pickle.dump(model, f)
ytest = np.array(y_test)
# confusion matrix and classification report(precision, recall, F1-score)
print(classification_report(ytest, model.predict(X_test)))
print(confusion_matrix(ytest, model.predict(X_test)))


# # test

# In[21]:


X_train.head(4)


# In[86]:


X_test1 =  test['ReviewText']


# In[72]:


X_test1.head(3)


# In[95]:


pred_test = model.predict(X_test1)


# In[96]:


pred_test


# In[97]:


xx = pred_test.tolist()


# In[98]:


df = pd.Series(xx)
df.head(2)


# In[100]:


submission = pd.DataFrame({'Review Text':test['ReviewText'], 'Review Title':test['ReviewTitle'],'topic':df})
submission.head(3)


# In[101]:


filename = 'submission_Amazon_lr.csv'

submission.to_csv(filename,index=False)
print('Saved file: ' + filename)

