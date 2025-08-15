import streamlit as st
import pandas as pd
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import requests
from bs4 import BeautifulSoup
from langdetect import detect
from deep_translator import GoogleTranslator
import joblib
import time

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def get_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        combined_text = ' '.join([p.get_text() for p in paragraphs])
        return combined_text.strip()
    except:
        return ""

def safe_translate(text, source_lang, target_lang, retries=3, delay=2):
    for _ in range(retries):
        try:
            return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        except Exception:
            time.sleep(delay)
    return None

vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("logistic_model.pkl")

st.title("Document Classification App")

uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
url_input = st.text_input("Or enter a URL:")
manual_text = st.text_area("Or write/paste your text here:")

text = None
if uploaded_file is not None:
    text = uploaded_file.read().decode('utf-8').strip()
elif url_input:
    text = get_text_from_url(url_input)
elif manual_text:
    text = manual_text.strip()

if text:
    if len(text) > 3:  
        lang = detect(text)
        if lang == "ar" and len(text) > 5:  
            translated = safe_translate(text, 'auto', 'en')
            if translated:
                text = translated
            else:
                st.error("Translation failed after multiple attempts.")
                text = None
        if text:
            clean = clean_text(text)
            features = vectorizer.transform([clean])
            prediction = model.predict(features)[0]
            confidence = model.predict_proba(features).max() * 100
            st.success(f"Predicted Class: {prediction}")
            st.info(f"Confidence Score: {confidence:.2f}%")
    else:
        st.error("Text is too short for processing.")
