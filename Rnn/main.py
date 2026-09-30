#Step 1: Import the necessary libraries and load the IMDB dataset
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing import sequence
word_index = imdb.get_word_index()
# Create a reverse mapping of the word index
reverse_word_index = {value: key for key, value in word_index.items()}
#Load the trained model
model = load_model('simplernn_model.h5')
#function to decode the review back to words
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

#function to predict the sentiment of a review
def predict_sentiment(review):
    # Preprocess the review
    encoded_review = [word_index.get(word, 2) + 3 for word in review.split()]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)

    # Make the prediction
    prediction = model.predict(padded_review)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    
    return sentiment, prediction[0][0] 

import streamlit as st
# Streamlit app
st.title("IMDB Movie Review Sentiment Analysis")
# Input text area for user to enter a review
st.write("Enter a movie review below and click 'Predict' to see the sentiment.")
if st.button("Predict"):
    review = st.text_area("Movie Review")
    if review:
        sentiment, confidence = predict_sentiment(review)
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Confidence: {confidence:.2f}")
    else:
        st.write("Please enter a review to analyze.")
        