### Libraries ###

# language detection
from langdetect import detect

# text preprocessing
import nltk
nltk.download('punkt')        
nltk.download('stopwords')    
nltk.download('wordnet')  

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
stop_words = set(stopwords.words('english'))

## Language Detection

def detect_language(text):
    try:
        lang = detect(text)
    except:
        lang = 'unknown'  
    return lang

## Text Preprocessing 

def stopword_remover(tokenized_comment, stop_words):
    clean_text = []
    for token in tokenized_comment:
        if token not in stop_words:
            clean_text.append(token)
    return clean_text

def preprocessor(reviews,
                         remove_punctuation=False,
                         lowercase=False,
                         tokenized_output=False,
                         remove_stopwords=True,
                         lemmatization=False,
                         stemming=False,
                         sentence_output=True):
    
    clean_text = reviews
    stop_words = set(stopwords.words('english'))
    
    # Punctuation removal
    if remove_punctuation:
        clean_text = re.compile(r'[^\w\s]').sub(' ', clean_text)
        
    # Lowercasing
    if lowercase:
        clean_text = clean_text.lower()
    
    # Tokenization
    clean_text = word_tokenize(str(clean_text))
    
    # Stopwords removal
    if remove_stopwords:
        clean_text = stopword_remover(clean_text, stop_words)
    
    # Lemmatization and Stemming
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        clean_text = [lemmatizer.lemmatize(token) for token in clean_text]
        
    elif stemming:
        stemmer = PorterStemmer()
        clean_text = [stemmer.stem(token) for token in clean_text]
        
    # Convert back to string if tokenized_output is False
    if not tokenized_output:
        clean_text = " ".join(clean_text)
        clean_text = re.sub(r'(\s)(?!\w)', '', clean_text)  # Remove space before punctuation
    
    # Sentence output
    if sentence_output:
        clean_text = sent_tokenize(str(clean_text))
    
    return clean_text

