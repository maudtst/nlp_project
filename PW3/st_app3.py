import streamlit
import os
import sys
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st
sys.path.insert(1,"CNN")
import loadCNN as cnn

articles, abstracts = cnn.loadCNN()
tfidf_vectorizer = TfidfVectorizer()
tfidf_articles  = tfidf_vectorizer.fit_transform(articles)
tfidf_abstracts = tfidf_vectorizer.transform(abstracts)


st.markdown(
    "<h1 style='text-align: center; background-color: #F63366; color: white;"
    "font-size: 32px; padding: 10px; border-radius: 10px;'>WP3 App</h1>",
    unsafe_allow_html=True,
)

request = st.text_input("What are you looking for?:")

if st.button("Find article"):
    if request:
        request = [request]
        tfidf_request = tfidf_vectorizer.transform(request)
        score_request = linear_kernel(tfidf_request,tfidf_articles)
        i = np.argmax(score_request[0])
        best_similarity_score = np.max(score_request[0])
        st.write(f"- Best similarity of resumé : {best_similarity_score}")
        st.write(f"- Article:\n\n {articles[i]}")