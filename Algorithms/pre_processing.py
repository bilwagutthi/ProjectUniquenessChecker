""" 
    This program defines a class of functions for different forms of text pre-processing.

    Some basic pre-processing such as:
    
        1. Remove HTML tags
        2. Remove extra whitespace
        3. Convert accented characters to ASCII characters
        4. Expand contractions
        5. Remove special characters/Punctuation 
        6. Lowercase all texts
    
    Some special functions such as
        7. Convert number words to numeric form
        8. Remove numbers
        9. Remove stopwords
        10. Lemmatization
        11. Tokanization   
"""

import inflect
import nltk
import string
import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize 

  
class pre_processing:

    def text_lowercase(text):
        return text.lower() 

    # Function to convert numbers to words
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
    
    # Function to remove punctuation
    def remove_punctuation(text): 
        translator = str.maketrans('', '', string.punctuation) 
        return text.translate(translator) 
    
    # Remove numbers 
    def remove_numbers(text):
        result = re.sub(r'\d+', '', text)
        return result 

    # remove whitespace from text 
    def remove_whitespace(text):
        return  " ".join(text.split())

    
    # remove stopwords function
    def remove_stopwords(text):
        stop_words = set(stopwords.words("english"))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in stop_words]
        return filtered_text 
 
    # stem words in the list of tokenised words 
    def stem_words(text):
        stemmer = PorterStemmer()
        word_tokens = word_tokenize(text) 
        stems = [stemmer.stem(word) for word in word_tokens] 
        return stems        

    
     
    # lemmatize string 
    def lemmatize_word(text):
        lemmatizer = WordNetLemmatizer()
        word_tokens = word_tokenize(text) 
        # provide context i.e. part-of-speech 
        lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in word_tokens] 
        return lemmas

    #remove html tags from text

    def strip_html_tags(text):
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text(separator=" ")
        return stripped_text

#remove accented characters from text, e.g. café
    def remove_accented_chars(text):
        text = unidecode.unidecode(text)
        return text             
# expand shortened words, e.g. don't to do not
    def expand_contractions(text):
       
        text = list(cont.expand_texts([text], precise=True))[0]
        return text    

    def create_tokens(text):
        doc = nlp(text)
        tokens = [w2n.word_to_num(token.text) if token.pos_ == 'NUM' else token for token in doc]
        return tokens

    def basic_ops(text):
        text=text_lowercase(text)
        text=remove_whitespace(text)
        text=remove_accented_chars(text)

        return text
    