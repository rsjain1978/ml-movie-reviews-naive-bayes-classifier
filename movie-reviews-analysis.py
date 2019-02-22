import pandas as pd

print ('Creating dataset')

#dataset downloaded from http://boston.lti.cs.cmu.edu/classes/95-865-K/HW/HW3/
df = pd.read_csv('movie-pang02-train.csv')

import re
from nltk.corpus import stopwords

def preProcessData(data):
    countMessages = data['reviews'].size

    processedReviews = []

    for i in range(0, countMessages):
        review = data['reviews'][i]
        letters_only = re.sub("[^a-zA-Z]",' ', review)
        words = letters_only.lower().split()
        stops = set(stopwords.words("english"))
        filtered_words = [w for w in words if not w in stops]
        processedReviews.append(" ".join(filtered_words))

    return processedReviews

print ('Pre-processing')
processedReview = preProcessData(df)

df["p_reviews"] = processedReview
columns = ['p_reviews','category']
df=df[columns]

from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder()
df['category'] = lc.fit_transform(df['category'])

#convert text into numbers
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

cv = CountVectorizer(analyzer='word')
train_reviews = cv.fit_transform(df['p_reviews'])
train_category = df['category']

tfidf_transformer = TfidfTransformer()
train_reviews_tfidf = tfidf_transformer.fit_transform(train_reviews)

print ('Training')

mb = MultinomialNB(fit_prior=True)
clf = mb.fit(train_reviews_tfidf, train_category)

testDF = pd.read_csv('movie-pang02-test.csv')
p_test_reviews = preProcessData(testDF)
testDF = pd.DataFrame(data=p_test_reviews, columns=['reviews'])
test_reviews_cv = cv.transform(testDF['reviews'])

print ('Predicting')

predictedY = clf.predict(test_reviews_cv)

for i in range(0, predictedY.size):
    if (predictedY[i]==0):
        print ('Bad Review')
    if (predictedY[i]==1):
        print ('Good Review')

