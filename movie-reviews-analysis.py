import pandas as pd

print ('Creating dataset')

#dataset downloaded from http://boston.lti.cs.cmu.edu/classes/95-865-K/HW/HW3/
df = pd.read_csv('movie-pang02-train.csv')

import re
from nltk.corpus import stopwords

print ('Pre-processing')

#convert text into numbers
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('vect', CountVectorizer(stop_words='english', lowercase=True)),
    ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True)),
    ('clf', MultinomialNB(alpha=1))

])

print ('Training')
pipeline.fit(df['reviews'], df['category'])

print ('Predicting')
testDF = pd.read_csv('movie-pang02-test.csv')
predictedY = pipeline.predict(testDF['reviews'])

for i in range(0, predictedY.size):
    if (predictedY[i]=='Neg'):
        print ('Bad Review')
    if (predictedY[i]=='Pos'):
        print ('Good Review')
