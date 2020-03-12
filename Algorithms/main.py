"""
    Main program to check uniqueness

"""

from Sample_Data.data import sample_data
from Sample_Data.test_data import title,abstract
from pre_processing import pre_processing
from k_Cluster_title import cosine_distence as cd


# Step 1 : Pre-Processing

pp=pre_processing()

# Pre processing of testing data
test_title_tokens = pp.advanced_ops(title)
test_abstract_tokens = pp.advanced_ops(abstract)

# Pre processing of sample data
sample_titles = sample_data.keys()
sample_title_tokens = []
for title in sample_titles:
    title_tokens = pp.advanced_ops(title)
    sample_title_tokens.append(title_tokens)

sample_abstracts = sample_data.values()
sample_abstracts_tokens = []
for abstracts in sample_abstracts:
    abstract_tokens = pp.advanced_ops(abstracts)
    sample_abstracts_tokens.append(abstract_tokens)

#print(test_title_tokens,'\n\n\n',test_abstract_tokens,'\n\n\n',sample_title_tokens[25],'\n\n\n',sample_abstracts_tokens[25])


# Step 2 : Uniqueness score

# Calculating cosine distance
title_dists=[]

for sample_title in sample_title_tokens:
    print("--------")
    print(sample_title)
    title_dist=cd.distence(test_title_tokens,sample_title)*100
    title_dists.append(title_dist)
    print(title_dist)

cosine_title_dist=sum(title_dists)/len(title_dists)


print(cosine_title_dist,title_dists)







