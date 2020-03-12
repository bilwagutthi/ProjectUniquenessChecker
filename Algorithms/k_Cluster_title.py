"""

    Program to measure distance between two sentences using cosine measurement

"""

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

class cosine_distence:

    def distence(test_title,compare_title):
        X_set=set(test_title)
        Y_set=set(compare_title)


        # Form a set containing keywords from both list of tokens
        l1=[]
        l2=[]
        rvector = X_set.union(Y_set)  
        for w in rvector: 
            if w in X_set:
                l1.append(1)
            else:
                l1.append(0) 
            if w in Y_set:
                l2.append(1) 
            else:
                l2.append(0) 
        
        # cosine formula
        c=0
        for i in range(len(rvector)): 
                c+= l1[i]*l2[i] 
        cosine = c / float((sum(l1)*sum(l2))**0.5) 
        
        return cosine
