import numpy as np
import pickle
import pandas as pd
import tensorflow as tf

import re
import ftfy
import fasttext

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Data Cleaning by removing specific words and symbols that have no meaning

# Lemitization of Text

def preprocess_data(text):
    text = ftfy.fix_text(text)
    text = text.lower()
    text = re.sub(r"received from:",' ',text)
    text = re.sub(r"from:",' ',text)
    text = re.sub(r"to:",' ',text)
    text = re.sub(r"subject:",' ',text)
    text = re.sub(r"sent:",' ',text)
    text = re.sub(r"ic:",' ',text)
    text = re.sub(r"cc:",' ',text)
    text = re.sub(r"bcc:",' ',text)
    # Remove email 
    text = re.sub(r'\S*@\S*\s?', '', text)
    # Remove numbers 
    text = re.sub(r'\d+','' ,text)
    # Remove new line characters 
    text = re.sub(r'\n',' ',text)
    # Remove hashtag while keeping hashtag text
    text = re.sub(r'#','', text)
    # Handle & symbol 
    text = re.sub(r'&;?', 'and',text)
    # Remove HTML special entities (e.g. &amp;)
    text = re.sub(r'\&\w*;', '', text)
    #remove html tags
    text = re.sub(r'<.*?>',' ',text)
    # Remove hyperlinks
    text = re.sub(r'https?:\/\/.*\/\w*', '', text)  
    #remove punctuation !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    # insert spaces between special characters to isolate them   
    special_char_pattern = re.compile(r'([{.(-):/\@#$%&;<=?>_!}])')
    text = special_char_pattern.sub(" \\1 ", text)
    text = text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
         
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    #remove more than 1 space
    text = re.sub(r'\s+',' ',text)
    text = text.strip()

    return text

