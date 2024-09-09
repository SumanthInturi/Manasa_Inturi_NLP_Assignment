import pandas as pd
import re
import unicodedata
import random
import spacy
from bs4 import BeautifulSoup
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
import contractions
from googletrans import Translator

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nlp = spacy.load('en_core_web_sm')

def clean_text(raw_text):
    text = BeautifulSoup(raw_text, "html.parser").get_text()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text

def replace_with_synonym(text, replacement_prob=0.1):
    words = text.split()
    new_words = []
    for word in words:
        if random.random() < replacement_prob:
            synonyms = wordnet.synsets(word)
            if synonyms:
                synonym = random.choice(synonyms).lemmas()[0].name()
                new_words.append(synonym)
            else:
                new_words.append(word)
        else:
            new_words.append(word)
    return ' '.join(new_words)

def back_translate(text, src='en', dest='de'):
    translator = Translator()
    translated = translator.translate(text, src=src, dest=dest).text
    back_translated = translator.translate(translated, src=dest, dest=src).text
    return back_translated

def replace_entities(text, new_entities):
    doc = nlp(text)
    new_text = text
    for ent in doc.ents:
        if ent.text in new_entities:
            new_text = new_text.replace(ent.text, new_entities[ent.text])
    return new_text

def normalize_unicode(text):
    return unicodedata.normalize('NFC', text)

def correct_spelling(text):
    blob = TextBlob(text)
    return str(blob.correct())

def segment_sentences(text):
    return sent_tokenize(text)

def tokenize_words(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.lower() not in stop_words]

def normalize_text(text):
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def stem_words(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

def lemmatize_words(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

csv_file_path = 'Dataset.csv'

df = pd.read_csv(csv_file_path)

def preprocess_text(text):
    cleaned_text = clean_text(text)
    synonym_text = replace_with_synonym(cleaned_text)
    back_translated_text = back_translate(synonym_text)
    new_entities = {'world': 'universe'}
    entity_replaced_text = replace_entities(back_translated_text, new_entities)
    normalized_text = normalize_unicode(entity_replaced_text)
    corrected_text = correct_spelling(normalized_text)
    sentences = segment_sentences(corrected_text)
    words = [tokenize_words(sentence) for sentence in sentences]
    words_no_stopwords = [remove_stopwords(word_list) for word_list in words]
    normalized_words = [normalize_text(' '.join(word_list)) for word_list in words_no_stopwords]
    stemmed_words = [stem_words(word_tokenize(text)) for text in normalized_words]
    lemmatized_words = [lemmatize_words(word_list) for word_list in stemmed_words]
    return lemmatized_words

df['preprocessed_text'] = df['text'].apply(preprocess_text)

print(df[['text', 'preprocessed_text']].head())

df.to_csv('processed_data.csv', index=False)
