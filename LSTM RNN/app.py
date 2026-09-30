import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load model
model = load_model("next_word_lstm.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)


# Function to predict next word
def predict_next_word(input_text):

    # Convert text to token numbers
    token_list = tokenizer.texts_to_sequences([input_text])[0]

    # Make sequence the same length used during training
    max_sequence_len = model.input_shape[1]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len,
        padding="pre"
    )

    # Predict probabilities for all words
    predicted_probs = model.predict(token_list, verbose=0)

    # Get word with highest probability
    predicted_index = np.argmax(predicted_probs, axis=-1)[0]

    # Convert number back to word
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word

    return None


# Streamlit UI
st.title("Next Word Prediction using LSTM")

st.write("Enter some text and the model will predict the next word.")

input_text = st.text_input(
    "Enter your sentence:",
    placeholder="For example: I am going"
)


if st.button("Predict Next Word"):

    if input_text.strip():

        next_word = predict_next_word(input_text)

        if next_word:
            st.success(f"Predicted next word: **{next_word}**")
        else:
            st.warning("Could not predict a word.")

    else:
        st.warning("Please enter some text.")