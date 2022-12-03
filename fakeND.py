#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


import nltk
nltk.download("stopwords")


# In[3]:


# printing the stopwords in english
print(stopwords.words("english"))


# In[6]:


news_dataset = pd.read_csv('train.csv')


# In[7]:


news_dataset.shape


# In[8]:


# print the first 5 rows of the dataframe
news_dataset.head()


# In[9]:


# counting the number of missing values in the dataset
news_dataset.isnull().sum()


# In[10]:


# replacing the null values with empty string
news_dataset = news_dataset.fillna('')


# In[11]:


# merging the author name and news title
news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']


# In[12]:


print(news_dataset['content'])


# In[13]:


# separating the data & label
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']


# In[14]:


print(X)
print(Y)


# In[ ]:


Stemming:

Stemming is the process of reducing a word to its Root word

example: actor, actress, acting --> act


# In[16]:


port_stem = PorterStemmer()


# In[17]:


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# In[18]:


news_dataset['content'] = news_dataset['content'].apply(stemming)


# In[19]:


print(news_dataset['content'])


# In[20]:


#separating the data and label
X = news_dataset['content'].values
Y = news_dataset['label'].values


# In[21]:


print(X)


# In[22]:


print(Y)


# In[23]:


# converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)


# In[24]:


print(X)


# In[25]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)


# In[26]:


model = LogisticRegression()


# In[27]:


model.fit(X_train, Y_train)


# In[28]:


# accuracy score on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[29]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[30]:


# accuracy score on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[31]:


print('Accuracy score of the test data : ', test_data_accuracy)


# In[40]:


X_new = X_test[10]

prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==0):
  print('The news is Real')
else:
  print('The news is Fake')


# In[38]:


print(Y_test[10])


# In[ ]:




