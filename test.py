"""
    This program is used to test for 5 different test cases in comparison to the database
        1 ) Title and abstract are both exactly the same
        2 ) Title is worded differently but abstract remains the same
        3 ) Title is the same but abstract is worded differently
        4 ) Title and abstract are both worded differently
        5 ) Title and abstract are completely different from the datasets in database 
"""

from main import main

# Dictionary with tiles as keys and abstracts as values

test_cases= {
    'One Quantifiable Security Evaluation Model for Cloud Computing Platform':'Whatever one public cloud, private cloud or a mixed cloud, the users lack of effective security quantifiable evaluation methods to grasp the security situation of its own information infrastructure on the whole. This paper provides a quantifiable security evaluation system for different clouds that can be accessed by consistent API. The evaluation system includes security scanning engine, security recovery engine, security quantifiable evaluation model, visual display module and etc. The security evaluation model composes of a set of evaluation elements corresponding different fields, such as computing, storage, network, maintenance, application security and etc. Each element is assigned a three tuple on vulnerabilities, score and repair method. The system adopts “One vote vetoed” mechanism for one field to count its score and adds up the summary as the total score, and to create one security view. We implement the quantifiable evaluation for different cloud users based on our G-Cloud platform. It shows the dynamic security scanning score for one or multiple clouds with visual graphs and guided users to modify configuration, improve operation and repair vulnerabilities, so as to improve the security of their cloud resources.',
    'Word2vec based Analysis of Similarity between Law Documents':'The similarity analysis of law documents is the basis of intelligent justice, while law documents based on several types of cases are quite different in terms of format and length, which causes trouble in analyzing similarities. For that we propose a more specific approach to dealing with law documents, combining Word2vec with legal documents corpus.',
    'The Collaborative Virtual Reality Neurorobotics Lab':'Presented in this paper is the collaborative Virtual Reality Neurorobotics Lab. This allows multiple collocated and far away users to experience, discuss and participate in neurorobotic experimentation in immersive VR(Virtual Reality). Described is  the coupling of the Neurorobotics Platform of the Human Brain Project with our collaboration of virtual reality and 3D telepresence infrastructure and future opportunities arising from the presented work  with simulated robots and brains based on researching common human interaction with robots',
    'Keyword extraction based on Dord2Vec weighted TextRank':'Research on the keyword extraction method of media blogs.On the basic idea of TextRank a candidate keywords graph model is built, useing Doc2Vec to determine the score of similarity between words as transition probability of node weight, determine the word score by loop method and pick the top N of the candidate keywords as result',
    'Current Twitter Trends Sentiment Analysis using Convolution Neural Network':'Twitter is a micro-blogging system that allows you to send and receive short posts called tweets. Twitter has become the best indicator of the wider pulse of the world and what is happening within it. Analyzing the nature of these tweets can be helpful in fields like business, sociology, economics, and physiological studies. Sentiment Analysis is the process of ‘computationally’ determining whether a piece of writing is positive, negative or neutral. When many Twitter users tweet about the same topics at the same time, they “trend” or become “trending topics”.Twitter then lists the current top 30 topics on the trending page. Topics break into the Trends list when the volume of Tweets about that topic at a given moment dramatically increases. Performing sentiment analysis on these topics helps to analyze whether the reaction to topic is positive or negative. This is done using a convolution neural network. The network is trained with a dataset which has tweets already labeled as positive and negative. Then the top tweets for current trending topics are gathered and passed through the neural network to determine whether the trend has more positive or negative reactions.'
}

test_descriptions=[
    "Title and abstract are both exactly the same",
    "Title is worded differently but abstract remains the same",
    "Title is the same but abstract is worded differently",
    "Title and abstract are both worded differently",
    "Title and abstract are completely different from the datasets in database"
]

print("\n\n","*"*100)
print("\n\t\t\tTESTING ON THE FOLLOWING CASES")
count=1

for key, value in test_cases.items():
    print("="*100)
    print("\n\n\n\tTest Case : ",count,"\n")
    print("\tTest Description : ",test_descriptions[count-1])
    count+=1
    print("\tTITLE : ",key,"\n")
#    print("\tABSTRACT : ",value,"\n")
    titlescore,abstractsscore=main(key,value)
    
    print("\n\t SIMILAR TITLES")
    i=1
    for result in titlescore:
        print("\n\tResult ",i,":")
        print("\n\t\tScore : ",result[0])
        print("\n\t\tTitle : ",result[1])
        i+=1
    
    print("\n\t SIMILAR ABSTRACTS")
    i=1
    for result in abstractsscore:
        print("\n\tResult ",i,":")
        print("\n\t\tScore : ",result[0])
        print("\n\t\tTitle : ",result[1])
        i+=1

print("\n\n\t\tTesting Finished\n\n","*"*100)
