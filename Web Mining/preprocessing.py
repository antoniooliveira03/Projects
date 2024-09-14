# Libraries

import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer
sent_tokenizer = PunktSentenceTokenizer()
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import pos_tag
from gensim.models import KeyedVectors
import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import emoji
import string
import random

plt.style.use('ggplot')

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Score Converter
    # Create new function to convert Rating scores to 3 categories
    # 1-4, 5-6, 7-10 forms negative, neutral, positive [0,1,2]

def score_convert_senti(score):
        if score <= 4:
            return 0
        elif score <= 6:
            return 1
        else:
            return 2

# Reviews Preprocessor

def stopword_remover(tokenized_comment, stop_words):
    clean_text = []
    for token in tokenized_comment:
        if token not in stop_words:
            clean_text.append(token)
    return clean_text

def reviews_preprocessor(reviews,
                 remove_punctuation = False,
                 lowercase = False,
                 tokenized_output = False,
                 remove_stopwords = True,
                 lemmatization = False,
                 stemming = False,
                 sentence_output = True):
    
    clean_text = reviews
    stop_words = set(stopwords.words('english'))
    
    
    
    # Punctuation
    if remove_punctuation:
        clean_text = re.compile(r'[^\w\s]').sub(' ', clean_text)
        
    # Lowercase    
    if lowercase:
        clean_text = clean_text.lower()
    
    #Tokenisation  
    clean_text = word_tokenize(str(clean_text))
    
    # Stopwords
    if remove_stopwords:
        clean_text = stopword_remover(clean_text, stop_words)
    
    # Lemmatisation and Stemming
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        clean_text = [lemmatizer.lemmatize(token) for token in clean_text]
        
    elif stemming:
        stemmer = PorterStemmer()
        clean_text = [stemmer.stem(token) for token in clean_text]
        
     # Removing Tokenisation    
    if tokenized_output == False:
        #re-join
        clean_text = " ".join(clean_text)
        #Remove space before punctuation
        clean_text = re.sub(r'(\s)(?!\w)','',clean_text)

    if sentence_output:
        clean_text = sent_tokenize(str(clean_text))
    
    
    return clean_text


## LSTM preprocessing
def lstm_preprocessing(dataset: pd.DataFrame, tokenizer=word_tokenize):
    random.seed(20)
    np.random.seed(20)

    #place reviews column textual data into list
    reviews = dataset["Reviews"]
    reviews_list = list(reviews)

    #Remove Punctuation
    reviews_list_noPunc = [_remove_punc(review) for review in reviews_list_deemojize]

    #Make text all lowercase
    reviews_list_lower = [review.lower() for review in reviews_list_noPunc]

    #Tokenization
    rev_tokenized = [tokenizer(review) for review in reviews_list_lower]

    # GloVe Vectorization
    rev_tokenized_embedded = _glove_embed(rev_tokenized)

    #Output
    ret_dataset = dataset.copy()
    assert type(ret_dataset) is pd.DataFrame
    ret_dataset["Tokenized_Reviews"] = rev_tokenized
    ret_dataset["Tokenized_Embedded_Reviews"] = rev_tokenized_embedded


    return ret_dataset



#internal function for removing punctuations
def _remove_punc(review):
    ascii_to_translate = str.maketrans("", "", string.punctuation)
    review = review.translate(ascii_to_translate)
    return review


## Function for glove embedding
def _glove_embed(tokenized_reviews):
    glove_model = api.load("glove-wiki-gigaword-50")

    rev_tokenized_embedded = []
    unidentified_tokens = [] ## Tokens not in GloVe model

    for review in tokenized_reviews:
        curr_embedded_review = []
        for token in review:
            if token in glove_model:
                curr_embedded_review.append(glove_model[token])
            else:
                unidentified_tokens.append(token)
        rev_tokenized_embedded.append(curr_embedded_review)
    print(f'{len(unidentified_tokens)} total tokens not in GloVe model: \n{unidentified_tokens}')

    return rev_tokenized_embedded

#preprocessing function for Logreg and Randforrest

def preprocess_reviews(text):
    # Convert text to lowercase
    text = text.astype(str).str.lower()
    
    # Remove emojis
    text = text.apply(lambda x: emoji.demojize(x))
    text = text.str.replace(r':[a-z_]+:', ' ', regex=True)
    
    # Remove special characters and numbers
    text = text.str.replace(r'[^a-zA-Z\s]', '', regex=True)
    
    # Tokenization (split the text into sentences)
    sentences = text.apply(lambda x: sent_tokenize(x))
    
    # Flatten list of sentences
    sentences = sentences.explode()
    
    # Tokenize sentences into words
    words = sentences.apply(lambda x: word_tokenize(x))
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = words.apply(lambda x: [word for word in x if word not in stop_words])
    
    # Stemming
    stemmer = PorterStemmer()
    stemmed_words = words.apply(lambda x: [stemmer.stem(word) for word in x])
    
    # Join the words back into a single string
    preprocessed_text = stemmed_words.apply(lambda x: ' '.join(x))
    
    return preprocessed_text
