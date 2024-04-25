import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import librosa





def theHighest(v1, v2, v3, v4):
    n = max(v1, v2, v3, v4)
    if v1 == n:
        return "Classical"
    elif v2 == n:
        return "Others"
    elif v3 == n:
        return "Reggaeton"
    elif v4 == n:
        return "Rock" 


st.header(":green[MUSIC GENRE CLASSIFIER]", divider="green")
st.subheader(":green[Predict Music Genre Through Audio Feature]")
st.write(":green[(Classical, Reggaeton, Rock, Others)]")
st.markdown("#")

#model=pickle.load(open('music_genre_classifier.pkl','rb'))




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

#'acousticness','danceability','energy','liveness','loudness','speechiness', 'tempo',	'valence'

# Define the sliders
acousticness = st.slider('Acousticness', min_value=0.0, max_value=1.0, value=0.5, step=0.001, format="%.2f")
danceability = st.slider('Danceability', min_value=0.0, max_value=1.0, value=0.5, step=0.001, format="%.3f")
energy = st.slider('Energy', min_value=0.0, max_value=1.0, value=0.5, step=0.0001, format="%.4f")
liveness = st.slider('Liveness', min_value=0.0, max_value=1.0, value=0.5, step=0.001, format="%.3f")
loudness = st.slider('Loudness', min_value=-100.0, max_value=5.0, value=-52.0, step=0.0001, format="%.4f")
speechiness = st.slider('Speechiness', min_value=0.0, max_value=1.0, value=0.5, step=0.001, format="%.3f")
tempo = st.slider('Tempo', min_value=0.0, max_value=250.0, value=125.0, step=0.001, format="%.3f")
valence = st.slider('Valence', min_value=0.0, max_value=1.0, value=0.5, step=0.001, format="%.3f")
 

feature = np.array([acousticness, danceability, energy, liveness, loudness, speechiness,tempo, valence])
feature = feature.reshape(1, -1)
# Button to trigger genre classification



    


    
if st.button(':green[Classify Genre]'):
    model = tf.keras.models.load_model("music_genre_classifier.h5")
    predict = model.predict(feature)
    prd = pd.DataFrame(predict)
    value =  theHighest(prd.at[0, 0], prd.at[0, 1], prd.at[0, 2], prd.at[0, 3])
    st.markdown("#")
    st.subheader(f'Predicted Genre: :green[{value}]')


# Context menu that updates dynamically based on slider inputs

 