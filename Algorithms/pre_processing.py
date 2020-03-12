""" 
    This program defines a class of functions for different forms of text pre-processing.

    Some basic pre-processing such as:


        1. Lowercase all texts
        2. Remove special characters/Punctuation
        3. Remove HTML tags
        4. Remove extra whitespace
        5. Convert accented characters to ASCII characters
        6. Expand contractions
       
    
    Some special functions such as
        7. Convert number words to numeric form
        8. Remove numbers
        9. Remove stopwords
        10. Stemming
        10. Lemmatization
        11. Tokanization   
"""

#import inflect
import nltk
import string
import re
import unidecode
# import contractions
from bs4 import BeautifulSoup
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize


  
class pre_processing:

    # 1. Convert to lowercase

    def text_lowercase(self,text):
        return text.lower()

    # 2. Function to remove punctuation
    def remove_punctuation(self,text): 
        translator = str.maketrans('', '', string.punctuation) 
        return text.translate(translator)

    # 3. Remove html tags from text

    def strip_html_tags(self,text):
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text(separator=" ")
        return stripped_text

    # 4. Remove whitespace from text 
    
    def remove_whitespace(self,text):
        return  " ".join(text.split())
    
    # 5. Remove accented characters from text, e.g. café

    def remove_accented_chars(self,text):
        text = unidecode.unidecode(text)
        return text             
    
    # 6. Expand shortened words, e.g. don't to do not
    
    def expand_contractions(self,text):
        text = list(contractions.expand_texts([text], precise=True))[0]
        return text    

    # 7.Function to convert numbers to words
    
    def convert_number(text):
        # Split the string into a list of words
        p = inflect.engine() 
        temp_str = text.split()
        new_string = []
        for word in temp_str: 
            #if word is a digit, convert the digit to numbers and append into the new_string list else append the word as is.
            if word.isdigit():
                temp = p.number_to_words(word)
                new_string.append(temp)
            else:
                new_string.append(word)

            #join the words of new_string to form a string
        temp_str = ' '.join(new_string)
        return temp_str 
    
    # 8. Remove numbers 
    
    def remove_numbers(text):
        result = re.sub(r'\d+', '', text)
        return result 

    # 9. Remove stopwords function

    def remove_stopwords(self,text):
        stop_words = set(stopwords.words("english"))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in stop_words]
        strg=" "
        return (strg.join(filtered_text)) 
 
    # 10. Stem words in the list of tokenised word Eg:- "writing" -> "write"
    def stem_words(self,word_list):
        stemmer = PorterStemmer() 
        stems = [stemmer.stem(word) for word in word_list] 
        return stems        

    
     
    # lemmatize string 
    def lemmatize_word(text):
        lemmatizer = WordNetLemmatizer()
        word_tokens = word_tokenize(text) 
        # provide context i.e. part-of-speech 
        lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in word_tokens] 
        return lemmas

    # Returns word toekens
    def create_word_tokens(self,text):
        return word_tokenize(text)

    # Returns Sentence tokens
    def create_sentence_tokens(para):
        return sent_tokenize(para)

    # Function returns a string after performing some basic operations

    def basic_ops(self,text):
        text=self.text_lowercase(text)
        text=self.remove_punctuation(text)
        text=self.strip_html_tags(text)
        text=self.remove_whitespace(text)
        text=self.remove_accented_chars(text)
       # text=self.expand_contractions(text)
        #text=self.create_word_tokens(text)
        return text

    # Function returns a list of word tokens after some advanced pre-processing

    def advanced_ops(self,text):
        text=self.basic_ops(text)
        filtered_string=self.remove_stopwords(text)
        token_list=self.create_word_tokens(filtered_string)
        token_list=self.stem_words(token_list)
        return token_list

    