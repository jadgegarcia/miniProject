import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf


#model=pickle.load(open('music_genre_classifier.pkl','rb'))
model = tf.keras.models.load_model("music_genre_classifier.h5")


st.header(":green[MUSIC GENRE CLASSIFIER]", divider="green")
st.subheader(":green[Predict Music Genre Through Audio Feature]")
st.markdown("#")

# Apply CSS styling to change the color of all sliders to green
st.markdown(''' 
            
            <style> div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"]{
                background-color: #1DB954; box-shadow: rgb(14 38 74 / 20%) 0px 0px 0px 0.2rem;}
            </style>
            <style> div.stSlider > div[data-baseweb="slider"] > div > div > div > div
                                { color: #1DB954; } 
            </style>

            
            ''', unsafe_allow_html=True)

# Define the sliders
acousticness = st.slider('Acousticness', min_value=0.0, max_value=1.0, value=0.5, step=0.001, format="%.3f")
danceability = st.slider('Danceability', min_value=0.0, max_value=1.0, value=0.5, step=0.001, format="%.3f")
energy = st.slider('Energy', min_value=0.0, max_value=1.0, value=0.5, step=0.01, format="%.3f")
liveness = st.slider('Liveness', min_value=0.0, max_value=1.0, value=0.5, step=0.001, format="%.3f")
speechiness = st.slider('Speechiness', min_value=0.0, max_value=1.0, value=0.5, step=0.001, format="%.3f")
valence = st.slider('Valence', min_value=0.0, max_value=1.0, value=0.5, step=0.001, format="%.3f")
 

feature = np.array([acousticness, danceability, energy, liveness, speechiness, valence])
feature = feature.reshape(1, -1)
# Button to trigger genre classification


def theHighest(v1, v2, v3):
    if (v1 > v2) and (v1 > v3):
        return "Classical"
    elif v2 > v3:
        return "Reggaeton"
    else:
        return "Rock" 
    


    
if st.button(':green[Classify Genre]'):
    predict = model.predict(feature)
    prd = pd.DataFrame(predict)
    value =  theHighest(prd.at[0, 0], prd.at[0, 1], prd.at[0, 2])
    st.markdown("#")
    st.subheader(f'Predicted Genre: :green[{value}]')


# Context menu that updates dynamically based on slider inputs

 