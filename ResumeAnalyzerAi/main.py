# Standard library imports
import os
import re
import string
import pickle
import collections
from collections import Counter
from itertools import chain
from os import listdir
from os.path import isfile, join
from io import StringIO

# Third-party imports
import pandas as pd
import numpy as np
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Gensim imports
import gensim
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # Fixed: 'sklern' -> 'sklearn'
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from joblib import dump, load

# SpaCy imports
import spacy
from spacy.matcher import PhraseMatcher

# Textract import
import textract

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Optional: Set up warnings or configurations
import warnings
warnings.filterwarnings('ignore')

def CleanResume(resumeText):
    resumeText = re.sub('https\S+\s*',' ', resumeText)
    resumeText = re.sub('RT|cc',' ',resumeText)
    resumeText = re.sub('#\S+',' ',resumeText)
    resumeText = re.sub(' [%s] ' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""),' ',resumeText)
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText)
    resumeText = re.sub('\s+',' ',resumeText)
    return resumeText
