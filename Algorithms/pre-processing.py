'''import the inflect library''' 
import inflect 
p = inflect.engine() 
  
'''convert number into words '''
def convert_number(text):
    '''split string into list of words '''
    temp_str = text.split()
    '''initialise empty list'''
    new_string = []
    
    for word in temp_str: 
         '''if word is a digit, convert the digit '''
         '''to numbers and append into the new_string list '''
         if word.isdigit():
             temp = p.number_to_words(word)
             new_string.append(temp)
             
         ''' append the word as it is '''
         ''' else:'''
         new_string.append(word)

         '''join the words of new_string to form a string'''
         temp_str = ' '.join(new_string)
         return temp_str 
  
input_str = 'There are three balls in this bag, and twelve in the other one.'
convert_number(input_str)


'''remove punctuation'''
def remove_punctuation(text): 
    translator = str.maketrans('', '', string.punctuation) 
    return text.translate(translator) 
  
input_str = "Hey, did you know that the summer break is coming? Amazing right !! It's only 5 more days !!"
remove_punctuation(input_str)