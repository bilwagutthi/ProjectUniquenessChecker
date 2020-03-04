'''  This program is for text pre-processing and we performed a series of steps under each component.
     Remove HTML tags
     Remove extra whitespaces
     Convert accented characters to ASCII characters
     Expand contractions
     Remove special characters
     Lowercase all texts
     Convert number words to numeric form
     Remove numbers
     Remove stopwords
     Lemmatization        '''


#import the inflect library 
import inflect
import nltk
import string
import re 
p = inflect.engine() 
  
class pre_processing:

    def text_lowercase(text):
        return text.lower() 

    #convert number into words
    def convert_number(text):
        #split string into list of words 
        temp_str = text.split()
        #initialise empty list
        new_string = []
        
        for word in temp_str: 
            #if word is a digit, convert the digit 
            #to numbers and append into the new_string list 
            if word.isdigit():
                temp = p.number_to_words(word)
                new_string.append(temp)
                
            # append the word as it is 
            else:
                new_string.append(word)

            #join the words of new_string to form a string
            temp_str = ' '.join(new_string)
            return temp_str 
    
     #remove punctuation
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

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
# remove stopwords function
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return filtered_text 

from nltk.stem.porter import PorterStemmer 
from nltk.tokenize import word_tokenize 
stemmer = PorterStemmer() 
# stem words in the list of tokenised words 
def stem_words(text): 
    word_tokens = word_tokenize(text) 
    stems = [stemmer.stem(word) for word in word_tokens] 
    return stems        

from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize 
lemmatizer = WordNetLemmatizer() 
# lemmatize string 
def lemmatize_word(text): 
    word_tokens = word_tokenize(text) 
    # provide context i.e. part-of-speech 
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in word_tokens] 
    return lemmas

def strip_html_tags(text):
    """remove html tags from text"""
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text

def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text             

def expand_contractions(text):
    """expand shortened words, e.g. don't to do not"""
    text = list(cont.expand_texts([text], precise=True))[0]
    return text    

    text = """three cups of coffee"""
    doc = nlp(text)
    tokens = [w2n.word_to_num(token.text) if token.pos_ == 'NUM' else token for token in doc]

   