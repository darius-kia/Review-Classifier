#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[2]:


import pandas as pd
import numpy as np
import re
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm


# # Read Data

# In[3]:


df = pd.read_json("./reviews_Amazon_Instant_Video_5.json", lines=True, dtype={'overall': np.int64}) # cast "overall" (rating) column to float)


# # Pre-process

# ## Class balance

# In[24]:


X = df['reviewText']
y = df['overall']
print(X.shape)


# In[25]:


mapped = []
for i in range(len(y)):
    if y[i] > 3:
        mapped.append((X[i], "POSITIVE"))
    elif y[i] < 3:
        mapped.append((X[i], "NEGATIVE"))
    # else:
    #     mapped.append((X[i], "NEUTRAL"))
        


# In[26]:


pos = list(filter(lambda i: i[1] == "POSITIVE", mapped))
# neu = list(filter(lambda i: i[1] == "NEUTRAL", mapped))
neg = list(filter(lambda i: i[1] == "NEGATIVE", mapped))
# print(len(pos), len(neu), len(neg))
X = list(map(lambda i: i[0], pos[:len(neg)] + neg))
y = list(map(lambda i: i[1], pos[:len(neg)] + neg))


# ## Vectorize

# In[27]:


def remove_nums(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text) # remove numbers
    text = re.sub(r'_', '', text) # remove underscores
    text = re.sub(r'[?,.!@#$%^&*()_+]', '', text)
    return text


# In[28]:


vectorizer = TfidfVectorizer(preprocessor=remove_nums, stop_words='english', ngram_range=(1,2))
# vectorizer = TfidfVectorizer(preprocessor=remove_nums, stop_words='english')
X_vectorized = vectorizer.fit_transform(X)


# ## Attribute Selection

# In[29]:


from sklearn.feature_selection import SelectKBest


# ### All attributes

# In[11]:


X_new = X_vectorized


# ### Chi-squared

# In[30]:


from sklearn.feature_selection import chi2
X_new = SelectKBest(chi2, k=20000).fit_transform(X_vectorized, y)


# ### ANOVA F-Value

# In[40]:


from sklearn.feature_selection import f_classif
X_new = SelectKBest(f_classif, k=20000).fit_transform(X_vectorized, y)


# ### False positive rate test

# In[50]:


from sklearn.feature_selection import SelectFpr
X_new = SelectFpr(alpha=0.05).fit_transform(X_vectorized, y)


# ## Train test split

# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X_new, y, train_size=0.67) # train test split with 2/3 train


# # Classify

# ## Naive Bayes

# In[52]:


from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train.toarray(), y_train) 


# ## Random Forest

# In[53]:


from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(max_depth=10)
rf_classifier.fit(X_train.toarray(), y_train)


# ## Neighbors

# In[54]:


from sklearn.neighbors import KNeighborsClassifier

n_classifier = KNeighborsClassifier(n_neighbors = 10)
n_classifier.fit(X_train.toarray(), y_train)


# ## Logistic Regression

# In[55]:


from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(random_state = 0)
lr_classifier.fit(X_train.toarray(), y_train)


# # Evaluate

# In[56]:


print(nb_classifier.score(X_test.toarray(), y_test))


# In[57]:


print(rf_classifier.score(X_test.toarray(), y_test))


# In[58]:


print(n_classifier.score(X_test.toarray(), y_test))


# In[59]:


print(lr_classifier.score(X_test.toarray(), y_test))

