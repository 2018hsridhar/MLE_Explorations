from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pandas as pd

'''
Input -> book chapters for subscriber base ( list of strs )
Goal -> ret extractive summary of text
Leverage NLTK and TfidfVectorization

ROUGE evals summary
    GTL = human summary ( vs AI ) 
    measure LCS btwen texts

https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
https://www.nltk.org/

'''


def tldr(text_to_summarize):
    # Part 1 : Clean and Structure the text ( with NLTK )
    sentences = sent_tokenize(text_to_summarize)

    # Initialize stopwords and lemmatizer
    stop_words = set(stopwords.words('english'))
    # lemmatizer = WordNetLemmatizer()

    cleaned_sentences = []
    for sent in sentences:
        words = word_tokenize(sent)                    # Tokenize
        words = [w.lower() for w in words]           # Lowercase
        words = [w for w in words if w not in stop_words]  # Remove stopwords
        words = [w for w in words if w.isalpha()]    # Remove punctuation/numbers
        # words = [lemmatizer.lemmatize(w) for w in words]  # Lemmatize
        cleaned_sentences.append(" ".join(words))    # Reconstruct sentence

    # Part 2 : Apply actual TF-IDF 
    # Rank the sentences for extractive summary :-)
    # Each sentence := a document
    # fit transform is global ( sensical scoring )
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform(cleaned_sentences)
    cleaned_sentence_scores = tfidf_matrix.sum(axis=1)  # Sum along columns for each row

    # notice list(zip(,..))) style
    sentence_with_scores = list(zip(cleaned_sentences, cleaned_sentence_scores))

    # sort ( sentence, sentence_ranks)
    # DESC score
    # top X percentage ( 50% original doc)
    sentence_with_scores.sort(key=lambda x: x[1], reverse=True)

    max_frac = 0.25 # bound here
    num_sentences_to_pick = max(1, int(len(cleaned_sentences) * max_frac))

    # how to improve the Rouge-L f-score?
    top_sentences = sentence_with_scores[:num_sentences_to_pick]
    just_sentences = [x[0] for x in top_sentences]
    executiveSummary = "".join(just_sentences)
    return executiveSummary
