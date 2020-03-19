'''
Program to generate word2vec vectors

ref:https://github.com/likejazz/Siamese-LSTM/blob/master/util.py

'''

import gensim
import numpy as np

class  Word2Vec:
        
    def get_vecs(self,token_list,embedding_dim=300):
        vocabs = {}
        vocabs_cnt = 0
        vocabs_not_w2v = {}
        vocabs_not_w2v_cnt = 0
        word2vec = gensim.models.word2vec.Word2Vec.load("w2vtest1.model").wv
        for word in token_list:
            if word not in word2vec.vocab:
                if word not in vocabs_not_w2v:
                    vocabs_not_w2v_cnt += 1
                    vocabs_not_w2v[word] = 1

            if word not in vocabs:
                vocabs_cnt += 1
                vocabs[word] = vocabs_cnt

        #Declare embedding matrix
        embeddings = 1 * np.random.randn(len(vocabs) + 1, embedding_dim)  
        embeddings[0] = 0  
        
        # Build the embedding matrix
        for word, index in vocabs.items():
            if word in word2vec.vocab:
                embeddings[index] = word2vec.word_vec(word)
        del word2vec

        return embeddings


    def train_word2vec_model(token_list):
        model = gensim.models.Word2Vec(token_list, size=300)
        model.train(token_list, total_examples=len(toekn_list), epochs=10)
        model.save('w2vtest1.model')

