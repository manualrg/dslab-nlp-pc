"""
Model configuration module for text classification using Naive Bayes.

This module:
- Defines label mappings (F/NF).
- Implements Spanish tokenization with stemming.
- Constructs and returns a scikit-learn pipeline that implements the champion model
"""
import string
import typing
import logging
from unidecode import unidecode
from sklearn.pipeline import Pipeline
# some vectorizer: from sklearn.feature_extraction.text import XXXVectorizer
# some model: from sklearn....<

import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import spacy


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Constants
TARGET_MAP = {"F": 0, "NF": 1}
TARGET_MAP_REV = {v: k for k, v in TARGET_MAP.items()}


# Ensure NLTK resources are downloaded
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# Models
nlp = spacy.load('es_core_news_lg')

# Tokenization and stemming
def tokenizer_stemmer_es(text) -> typing.List[str]:
    stopword_es = nltk.corpus.stopwords.words('spanish')
    stemmer = SnowballStemmer("spanish")

    clean_words = [word for word in word_tokenize(text) if word not in string.punctuation and word.lower() not in stopword_es] # list[str]
    return [stemmer.stem(word) for word in clean_words]  # list[str]
    

def tokenizer_lemma_es(text,
                       max_input_len=nlp.max_length,  # 1000000
                       min_token_len= 2,
                       ) -> typing.List[str]:
    doc = nlp(text[:max_input_len])  # truncar el documento de entrada al máximo proporcionado por el modelo de spacy
    lemmas = [unidecode(token.lemma_) for token in doc if token.is_alpha
              and len(token) > min_token_len
              and not token.is_stop
              and not token.like_email
              and not token.like_url
              and not token.is_currency
              and token.ent_type_ not in ['PER', 'LOC', 'ORG']
              ]
    return lemmas # list[str]
    return stopwords_tok


def get_model(
    # your Hiperparameters:
    min_df: int = 3,
    max_df: float = 0.5,
    ...
):
    """
    Builds and returns a scikit-learn Pipeline for Spanish text classification

    Args:
        min_df (int): Minimum document frequency for vectorizer.
        max_df (float): Maximum document frequency for vectorizer.
        max_features (int, optional): Maximum number of features to include.

    Returns:
        sklearn.pipeline.Pipeline: A pipeline with vectorizer and a classifier.
    """
    
    logging.info("Building pipeline...")
    
    # Your vectorization strategy
    dtm_transformer = XXXVectorizer(
        ...
    )

    # Your model
    clf = ...


    # Champion pipeline architecture
    skl_pl = Pipeline([
        ('fte', dtm_transformer),
        ('clf', clf)
    ])

    return skl_pl

