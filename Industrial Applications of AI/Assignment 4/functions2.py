### Libraries ###

import pandas as pd

# Outliers Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Language detection
from langdetect import detect

# Text Translation
import requests

# Emoji Detection
import unicodedata
import emoji

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

## Outliers Visualisation
def plot_boxplots(data, columns_to_check, palette=None):
    for column in columns_to_check:
        plt.figure(figsize=(6, 4))  
        sns.boxplot(x=data[column], palette=palette) 
        plt.title(f'Box Plot of {column}')
        plt.xticks(rotation=45) 
        plt.show()


## Outliers Detection
def detect_outliers_per_column(data, columns_to_check):
    outlier_info = {}
    
    for column in columns_to_check:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        
        outlier_info[column] = {
            'Q1': Q1,
            'Q3': Q3,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count': len(outliers) 
        }
    
    return outlier_info


## Language Detection
def detect_language(text):
    try:
        lang = detect(text)
    except:
        lang = 'unknown'  
    return lang


## Text Translation
def translate_with_deepl(text, auth_key, source_lang, target_lang):
    detected_lang = detect_language(text)
    if detected_lang.lower() == target_lang.lower():
        print('Skipped')
        return text  # Text is already in the target language
    
    url = "https://api-free.deepl.com/v2/translate"
    params = {
        "auth_key": auth_key,
        "text": text,
        "source_lang": source_lang,
        "target_lang": target_lang
    }
    response = requests.post(url, data=params)
    if response.status_code == 200:
        translation = response.json()["translations"][0]["text"]
        print('Translated')
        return translation
    else:
        print(f"Translation failed with status code {response.status_code}")
        return None

## Emoji Detection
def is_emoji(character):
    return character in emoji.EMOJI_DATA

def contains_emoji(text):
    return any(is_emoji(char) for char in text)


## Emoji Replacement
def replace_emojis(text):
    result = []
    for char in text:
        if is_emoji(char):
            description = emoji.demojize(char)
            print(f"Replacing {char} with {description}")
            result.append(description[1:-1])  # Remove surrounding colons from emoji name
        else:
            result.append(char)
    return ''.join(result)


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

