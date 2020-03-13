''' 
  Algorithm for Doc2Vec
  ref:https://medium.com/@mishra.thedeepak/doc2vec-in-a-simple-way-fa80bfe81104     
'''

#Import all the dependencies
import gensim
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, join

tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('english'))
#This function does all cleaning of data using two objects above
def nlp_clean(data):
   new_data = []
   for d in data:
      new_str = d.lower()
      dlist = tokenizer.tokenize(new_str)
      dlist = list(set(dlist).difference(stopword_set))
      new_data.append(dlist)
   return new_data


   class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield gensim.models.doc2vec.LabeledSentence(doc,[self.labels_list[idx]])
              
              #iterator returned over all documents
              it = LabeledLineSentence(data, docLabels)
              model = gensim.models.Doc2Vec(size=300, min_count=0, alpha=0.025, min_alpha=0.025)
              model.build_vocab(it)
              #training of model
              for epoch in range(100):
                  print ('iteration '+str(epoch+1))
                  model.train(it)
                  model.alpha -= 0.002
                  model.min_alpha = model.alpha
                  #saving the created model
                  model.save('doc2vec.model')
                  print ('model saved')
                  #loading the model
                  d2v_model = gensim.models.doc2vec.Doc2Vec.load('doc2vec.model')
                  #start testing
                  #printing the vector of document at index 1 in docLabels
                  docvec = d2v_model.docvecs[1]
                  print ('docvec')
                  #printing the vector of the file using its name
                  docvec = d2v_model.docvecs['1.txt'] #if string tag used in training
                  print ('docvec')
                  #to get most similar document with similarity scores using document-index
                  similar_doc = d2v_model.docvecs.most_similar(14) 
                  print ('similar_doc')
                  #to get most similar document with similarity scores using document- name
                  sims = d2v_model.docvecs.most_similar('1.txt')
                  print ('sims')
                  #to get vector of document that are not present in corpus 
                  docvec = d2v_model.docvecs.infer_vector('war.txt')
                  print ('docvec')

def build_lda(trainingdata, vocabdict, topics=40, random_state=0):
    vectorizer = CountVectorizer(analyzer="word", vocabulary=vocabdict)
    trainingvectors = vectorizer.transform(trainingdata)
    lda_model = LatentDirichletAllocation(n_topics=topics,random_state=random_state)
    lda_model.fit(trainingvectors)
    return lda_model
