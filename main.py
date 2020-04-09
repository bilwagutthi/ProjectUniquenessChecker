"""
    This program returns the uniqueness.

"""

from Algorithms.Sample_Data.data import sample_data 
#from Sample_Data.test_data import title,abstract
from Algorithms.pre_processing import pre_processing
from Algorithms.k_Cluster_title import cosine_distence as cd
from Algorithms.tf_idf import TermFrequency as tf
from Algorithms.word2vec import Word2Vec
from Algorithms.lstm import Siamese_LSTM
import pandas as pd



def main(title,abstract):
    pp=pre_processing()
    test_title_tokens = pp.advanced_ops(title)
    test_abstract_tokens = pp.advanced_ops(abstract)

    dataset_titles = sample_data.keys()
    titles_dist= []
    for title in dataset_titles:
        title_token = pp.advanced_ops(title)
        title_dist=cd.distence(title_token,test_title_tokens)*100
        titles_dist.append(title_dist)
        
    top3titles=sorted(zip(titles_dist,dataset_titles),reverse=True)[:3]

    dataset_abstracts = sample_data.values()
    abstracts_dist = []
    wtv=Word2Vec()
    sml=Siamese_LSTM()
    for abstract in dataset_abstracts:
        abstract_token = pp.advanced_ops(abstract)
        dataframe_var={'sentences1':[test_abstract_tokens],'sentences2':[abstract_token]}
        df=pd.DataFrame(dataframe_var)
        train_df,embeddings=wtv.get_df_embs(df,embedding_dim=300)
        dist=sml.compare(train_df)
        abstracts_dist.append(dist[0][0]*100)
        
    top3abstracts=sorted(zip(abstracts_dist,dataset_titles),reverse=True)[:3]

    max_sim=max(max(top3abstracts),max(top3titles))
    uniqueness=100- max_sim[0]
    top_title=max_sim[1]

    return uniqueness,top_title,top3titles,top3abstracts

