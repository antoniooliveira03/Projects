################ Imports ################

# pandas and numpy 
import pandas as pd
import numpy as np

# regex
import regex as re

# Preprocessing
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer
sent_tokenizer = PunktSentenceTokenizer()
import unicodedata
lemmatizer = WordNetLemmatizer()
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
import ast
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from wordcloud import WordCloud

# Vectorization
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Visualization


################ Functions ################

stop_words = set(stopwords.words('english'))

def stopword_remover(tokenized_comment,stop_words = stop_words):
    clean_comment = []
    for token in tokenized_comment:
        if token not in stop_words:
            clean_comment.append(token)
    return clean_comment

def preprocessor(raw_text, 
                 lowercase=True, 
                 leave_punctuation = False, 
                 remove_stopwords = True,
                 stop_words = None,
                 correct_spelling = False, 
                 lemmatization=False, 
                 porter_stemming=False,
                 tokenized_output=False, 
                 sentence_output=False
                 ):
    

    clean_text = raw_text

    if lowercase == True:
        
    #convert to lowercase
        if any(ord(char) > 127 for char in clean_text):
            clean_text =  ''.join([unicodedata.normalize('NFKD', char).lower() for char in clean_text])
        else:
            clean_text = clean_text.lower()
        
    #remove newline characters
    clean_text = re.sub(r'(\**\\[nrt]|</ul>)',' ',clean_text) 
    
    #remove between []
    clean_text = re.sub(r'\[(.*?)\]', ' ', clean_text)
    
    if leave_punctuation == False:    
    #remove punctuation:
        clean_text = re.compile(r'[^\w\s]').sub(' ', clean_text)
        #clean_text = re.sub(r'([\.\,\;\?\!\:\'])',' ',clean_text) acho que o de cima funciona melhor pq retira { e (
        
    #remove url:
    clean_text = re.sub(r'(\bhttp[^\s]+\b)',' ',clean_text)
    
    #remove isolated consonants:
    clean_text = re.sub(r'\b([^aeiou-])\b',' ',clean_text)

    #remove words with 3 or more of the same letter in a row  aaaaaaaaaaaaahhhh
    clean_text = re.sub(r'\b(?:\w*(\w)\1{2,}\w*)\b', ' ',clean_text)

    #remove non-latin characters and numbers e.g. 好12
    clean_text = re.sub(r'[^a-zA-Z\s]', ' ', clean_text)
    
    #correct spelling
    if correct_spelling == True:
        incorrect_text = TextBlob(clean_text)
        clean_text = incorrect_text.correct()
        
    #tokenize
    clean_text = word_tokenize(str(clean_text))
    
    #remove stopwords
    if remove_stopwords == True:
    	clean_text = stopword_remover(clean_text,stop_words)
        
    #lemmatize
    if lemmatization == True:
        for pos_tag in ["v","n","a"]:
            clean_text = [lemmatizer.lemmatize(token, pos=pos_tag) for token in clean_text]
            
    elif porter_stemming == True:  
        porter_stemmer = PorterStemmer()
        clean_text = [porter_stemmer.stem(token) for token in clean_text]
    
    if tokenized_output == False:
    #re-join
        clean_text = " ".join(clean_text)
    #Remove space before punctuation
        clean_text = re.sub(r'(\s)(?!\w)','',clean_text)

    if sentence_output == True:
        #split into sentences:
        clean_text = sent_tokenizer.tokenize(str(clean_text))
    
    return clean_text

def read_corpus(column, tokens_only=False):
    for i, tokens in enumerate(column):
        try:
            tokens = ast.literal_eval(tokens)
        except:
            tokens = tokens
        if tokens_only:
            yield tokens
        else:
            yield TaggedDocument(tokens, [i])


def top_words_per_tag(dataset, column, label, k=100, bow=True):
    print(f'Computing {label}, k = {k}, column = {column}')
    # Filter the dataset based on the specified label
    corpus = dataset[column].loc[dataset["tag"] == label]
    # corpus = [' '.join(tokens) for tokens in corpus]
    
    # Choose the vectorizer based on the 'bow' parameter
    if bow:
        vectorizer = CountVectorizer(stop_words='english')
    else:
        vectorizer = TfidfVectorizer(stop_words='english')

    # Convert the tokenized text into a matrix of features
    X = vectorizer.fit_transform(corpus)

    # Get the feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Compute the column sums
    word_sums = X.sum(axis=0).A1
    
    # Sort columns by their sums in descending order
    sorted_words = sorted(zip(feature_names, word_sums), key=lambda x: x[1], reverse=True)
    
    # Select the top k columns
    top_words = [word for word, _ in sorted_words[:k]]
    
    print(top_words)
    return top_words


# Histogram
def histogram(data):
    plt.hist(data, bins=22, color='#FF914D', ec='#F16007', alpha=0.7, label='Data Points')  # Adjust the number of bins
    plt.title('Histogram')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.legend()

    # Set x-axis limits to include a larger range
    plt.xlim(min(data), max(data)) 

    plt.show()

# WordClouds
def word_cloud_generator(dataset, labels):
    methods = ["bow", "tfidf"]

    for method in methods:
        for label in labels:
            # Combine the lists of clean lyrics into a single string
            corpus = [' '.join(map(str, lyrics)) for lyrics in dataset["clean_lyrics"].loc[dataset["tag"] == label]]

            if method == "bow":
                vectorizer = CountVectorizer()
            else:
                vectorizer = TfidfVectorizer()

            fitted_model = vectorizer.fit_transform(corpus)

            top_freqs = dict(zip(vectorizer.get_feature_names_out(), np.ravel(fitted_model.sum(axis=0)).tolist()))

            wc = WordCloud(background_color="white", max_words=120, width=1000, height=500, color_func=lambda *args, **kwargs: (0, 0, 0))
            wc.generate_from_frequencies(top_freqs)
            wc.to_file("wordclouds/wc_{}_{}.png".format(method, label))

    
    
print('The functions.py file was imported successfully')

