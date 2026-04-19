import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


#LOad the ImDB dataset word index
word_index=imdb.get_word_index()
reserve_word_index ={value: key for key, value in word_index.items()}

#Load the pretrained model with Relu activation

model= load_model('simple_rnn_imdb.h5')

## Step 2 Helper Functions
# Function decode the reviews

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3,'?') for i in encoded_review])

#Function to preprocess user input
##def preprocess_text(text):
  ##  words=text.lower().split()
    ##encoded_review = [word_index.get(word, 2) + 3 for word in words]
 ##   padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
   ## return padded_review
    
def preprocess_text(text):
    words = text.lower().split()
    
    encoded_review = []
    for word in words:
        index = word_index.get(word, 2)  # 2 = OOV
        if index >= 10000:   # 🔥 limit to vocab size
            index = 2        # treat as unknown
        encoded_review.append(index + 3)

    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review    



##Step 3

##Prediction Function

def predict_sentiment(review):
    preprocess_input=preprocess_text(review)
    
    prediction = model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0]>0.5 else 'Negetive'
    return sentiment, prediction[0][0]


## Streamlit app


import streamlit as st
st.title("IMDB Movie Review Setiment Analysis")
st.write('Enter the movie review to classify it as the positive or negetive')

#User Input

user_input= st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input=preprocess_text(user_input)


    ##Make Prediction
    prediction=model.predict(preprocessed_input)
    setiments='Positive' if prediction[0][0]>0.5 else 'Negative'

    ## Display the Result

    st.write(f'Setiment: {setiments}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write(f'Please enter a movie Review')