import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

tfidf=pickle.load(open('tfidf.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return ' '.join(y)

st.title('Email or SMS Spam Detector')
input_text=st.text_area('Enter The Email/SMS')
transformed=transform_text(input_text)
vectorized=tfidf.transform([transformed])
prediction=model.predict(vectorized)[0]
if st.button('Predict'):
    if prediction == 0:
        st.header('This is not a spam')
    else:
        st.header('This is a spam')