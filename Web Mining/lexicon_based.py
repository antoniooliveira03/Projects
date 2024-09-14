from nltk.sentiment import SentimentIntensityAnalyzer
vader = SentimentIntensityAnalyzer()
from textblob import TextBlob
import matplotlib.pyplot as plt

# VADER 
def vader_algorithm(review, compound = True):

    # Get the polarity scores (negative, neutral and positive) for the review
    polarity_scores_ = vader.polarity_scores(review)
    
    # compound scores
    if compound:
        polarity = polarity_scores_["compound"]

    else:
        # The three separated scores
        polarity = polarity_scores_

    return polarity

# TextBlob
def textblob_sa(review):

    # Get the polarity (compounded polarity scores) for the song
    polarity = TextBlob(review).sentiment.polarity

    return polarity

def textblob_subjectivity(review):
    blob = TextBlob(str(review))
    return blob.sentiment.subjectivity


# Histograms
def plot_sentiment_histograms(dataset, sorted_sentiment, polarity_column, color='steelblue'):
    # Define the number of rows and columns for subplots
    num_rows = 1
    num_cols = len(sorted_sentiment)

    # Create subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5))

    # Loop through sorted unique sentiments and corresponding subplot axes
    for i, sentiment_ in enumerate(sorted_sentiment):
        # Plot histogram for each sentiment
        axes[i].hist(dataset[dataset['Sentiment'] == sentiment_][polarity_column])

        # Set subplot title
        axes[i].set_title(f'Sentiment: {sentiment_}')

        # Set labels for the x and y axes
        axes[i].set_xlabel(polarity_column)
        axes[i].set_ylabel('Frequency')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

