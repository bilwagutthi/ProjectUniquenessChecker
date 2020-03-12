"""

    Program for UESTS


"""

from numpy import *

class uests:

    def align(s1,s2,threshold):
        s1len=len(s1)
        s2len=len(s2)
        M=zeros((s2len,s1len))
        print(M)

        for i in range(s1len):
            for  j in range(s2len):
                
                M[j]=1   
        
        print(M)

sample=uests
uests.align(['i','happy','today'],['sad','tomorrow'],0.5)
