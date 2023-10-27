import nltk
from nltk.tag import UnigramTagger
from nltk.corpus import treebank
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.sentiment.util import mark_negation

nltk.download('wordnet')
nltk.download('sentiwordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('treebank')
nltk.download("punkt")
  
import streamlit as st
import os
import sys
import numpy as np

tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
tagged_sents = nltk.corpus.treebank.tagged_sents()
unigram_tagger = UnigramTagger(tagged_sents)

#Identification des adejctifs / adverbes
def find_tag(tagged_sentence,tag_name="JJ"):
    words = [word for word, tag in tagged_sentence if tag and tag.startswith(tag_name)]
    return words

def classWords(review,tokenizer,unigram_tagger,tag_name="JJ"):
    sentences = tokenizer.tokenize(review) #tokenize chaque critique en liste de phrases
    word_class=[]
    for sentence in sentences:
        words = word_tokenize(sentence)                     # Tokenize chaque phrase liste de mots
        tagged_words = unigram_tagger.tag(words)            # Etiquetage
        word_class.extend(find_tag(tagged_words,tag_name))  # Identifie les adverbes
    return word_class


def getnewTag(tag_name):
    sentiwordnet_pos = None
    if tag_name.startswith("JJ"):   # Adjectif
        sentiwordnet_pos = 'a'
    elif tag_name.startswith("RB"): # adverb
        sentiwordnet_pos = 'r'
    elif tag_name.startswith("V"):  # Verb
        sentiwordnet_pos = 'v'
    elif tag_name.startswith("NN"): #Noun
        sentiwordnet_pos = 'n'
    return sentiwordnet_pos

def getSentiment(tagged_word,tag_name):
    tag_name = getnewTag(tag_name)
    if tag_name:
        synsets = list(swn.senti_synsets(tagged_word, tag_name))
        if synsets:
            synset = synsets[0]
            return synset.pos_score() - synset.neg_score()
    return 0

def getReviewScore(review,tag_names):
    score = 0
    count = 0
    
    for tag in tag_names:
        words = classWords(review,tokenizer,unigram_tagger,tag)
        score += np.sum([getSentiment(w,tag) for w in words])
        count += len(words)

    if count > 0:
        score = score / count
    return score

def classifyReview(review, tag_names, threshold=0.1):
    score = getReviewScore(review, tag_names)
    if score < 0 : return "Negative"
    else: return 'Positive'

st.markdown(
    "<h1 style='text-align: center; background-color: #F63366; color: white;"
    "font-size: 32px; padding: 10px; border-radius: 10px;'>WP2 App</h1>",
    unsafe_allow_html=True,
)

user_review = st.text_input("Enter you review:")

if st.button("Evaluate sentiment"):
    if user_review:
        sentiment = classifyReview(user_review,["RB","JJ"])
        st.write(sentiment)
