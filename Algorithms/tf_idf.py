from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import nltk

class TermFrequency:

    def result(test_list,sample_list):
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(sample_list)

        trainVectorizerArray = vectorizer.fit_transform(train_set).toarray()
        testVectorizerArray = vectorizer.transform(test_set).toarray()
        print ('Fit Vectorizer to train set', trainVectorizerArray)
        print ('Transform Vectorizer to test set', testVectorizerArray)


        print (transformer.fit(trainVectorizerArray))
        print (transformer.transform(trainVectorizerArray).toarray())

        print (transformer.fit(testVectorizerArray))
        
        tfidf = transformer.transform(testVectorizerArray)
        print (tfidf.todense())
