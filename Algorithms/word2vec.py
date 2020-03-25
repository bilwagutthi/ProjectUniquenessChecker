'''
Program to generate word2vec vectors

ref:https://github.com/likejazz/Siamese-LSTM/blob/master/util.py

'''

import gensim
import numpy as np
from pre_processing import pre_processing

class  Word2Vec:
    def trained_vecs(self,abstracts_tokens,embedding_dim):
        vocabs = {}
        vocabs_cnt = 0
        vocabs_not_w2v = {}
        vocabs_not_w2v_cnt = 0
        word2vec=gensim.models.word2vec.Word2Vec.load("w2vtest1.model").wv

        for token_list in abstracts_tokens:
            for word in token_list:
                if word not in word2vec.vocab:
                    if word not in vocabs_not_w2v:
                        vocabs_not_w2v_cnt += 1
                        vocabs_not_w2v[word] = 1
                    if word not in vocabs:
                        vocabs_cnt += 1
                        vocabs[word] = vocabs_cnt

        embeddings = 1 * np.random.randn(len(vocabs) + 1, embedding_dim) 
        embeddings[0] = 0 
        for word, index in vocabs.items():
            if word in word2vec.vocab:
                embeddings[index] = word2vec.word_vec(word)
        del word2vec

        return embeddings

    def get_df_embs(self,df,embedding_dim):
        vocabs = {}
        vocabs_cnt = 0
        vocabs_not_w2v = {}
        vocabs_not_w2v_cnt = 0
        word2vec = gensim.models.word2vec.Word2Vec.load("w2vtest1.model").wv
        pp=pre_processing()
        df['sentences1_n']=[[] for _ in range(len(df))]
        df['sentences2_n']=[[] for _ in range(len(df))]
        for index, row in df.iterrows():
            for sentence in ['sentences1', 'sentences2']:
                sent2 = []
                word_list=pp.advanced_ops(row[sentence])
                for word in word_list:
                    # If a word is missing from word2vec model.
                    if word not in word2vec.vocab:
                        if word not in vocabs_not_w2v:
                            vocabs_not_w2v_cnt += 1
                            vocabs_not_w2v[word] = 1

                    # If you have never seen a word, append it to vocab dictionary.
                    if word not in vocabs:
                        vocabs_cnt += 1
                        vocabs[word] = vocabs_cnt
                        sent2.append(vocabs_cnt)
                    else:
                        sent2.append(vocabs[word])
                df.at[index, sentence + '_n'] = sent2
        embeddings = 1 * np.random.randn(len(vocabs) + 1, embedding_dim)
        embeddings[0] = 0

        for word, index in vocabs.items():
            if word in word2vec.vocab:
                embeddings[index] = word2vec.word_vec(word)
        del word2vec

        return df, embeddings

    def train_word2vec_model(token_list):
        model = gensim.models.Word2Vec(token_list, size=300)
        model.train(token_list, total_examples=len(token_list), epochs=10)
        model_name='w2vtest1.model'
        model.save(model_name)
        return model_name

