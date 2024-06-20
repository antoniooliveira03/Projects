## Libraries 

# Vader
from nltk.sentiment import SentimentIntensityAnalyzer

# TexBlob
from textblob import TextBlob

# SentiWordNet
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn

# Afinn
from afinn import Afinn


##########

## Vader
vader = SentimentIntensityAnalyzer()

def vader_sa(review, compound = True):

    # Get the polarity scores (negative, neutral and positive) for the song
    polarity_scores_ = vader.polarity_scores(review)
    
    # If you want to compound the scores into one final score
    if compound:
        polarity = polarity_scores_["compound"]

    # If you want the three scores
    else:
        # The three separated scores
        polarity = polarity_scores_

    return polarity

## Textblob
def textblob_sa(review):

    # Get the polarity (compounded polarity scores) for the song
    polarity = TextBlob(review).sentiment.polarity

    return polarity

## SentiWordNet
def nltk_to_wordnet_pos(nltk_pos):
    if nltk_pos.startswith('N'):
        return wn.NOUN
    elif nltk_pos.startswith('V'):
        return wn.VERB
    elif nltk_pos.startswith('R'):
        return wn.ADV
    elif nltk_pos.startswith('J'):
        return wn.ADJ
    else:
        return None

def get_sentiment_score(word, pos_tag):
    try:
        synsets = list(swn.senti_synsets(word, pos=pos_tag))
        if synsets:
            return synsets[0].pos_score() - synsets[0].neg_score()
        else:
            return 0
    except KeyError:
        return 0
    

afinn = Afinn()