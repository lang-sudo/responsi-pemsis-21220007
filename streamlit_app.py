import streamlit as st
import pickle
import numpy as np
import json
import pandas as pd

# Load mapping dari features.json
with open('model/encode_dict.json', 'r') as file:
    encode_dict = json.load(file)

with open('model/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('model/lstm_model.pkl', 'rb') as file:
    lstm_model = pickle.load(file)

with open('model/svm_classifier.pkl', 'rb') as file:
    svm_classifier = pickle.load(file)

# Hapus objek `class` dari unique_values
encode_dict.pop('class', None)

# Judul Streamlit
st.title('Prediksi apakah Jamur ini bisa dimakan? 🍲😋⁉️')

# Menampilkan setiap fitur dengan opsi dan nilai numerik
input_features = []
for feature, options in encode_dict.items():
    selected_option = st.selectbox(f'{feature}', list(options.keys()))
    numeric_value = options[selected_option]
    input_features.append(numeric_value)
    st.write(f'{feature}: {selected_option} -> {numeric_value}')

df = pd.DataFrame([input_features])

def prediction(df):
    # Skala fitur numerik menggunakan Scaler yang dimuat
    scaled_data = scaler.transform(df)

    # Bentuk ulang input untuk LSTM
    input_data = scaled_data.reshape((scaled_data.shape[0], 1, scaled_data.shape[1]))

    # Ekstrak fitur menggunakan model LSTM
    features = lstm_model.predict(input_data)

    # Lakukan prediksi menggunakan model SVM
    prediction = svm_classifier.predict(features)

    if prediction:
        prediction = 'beracun ☠️, jangan ya dek ya 🙅'
    else:
        prediction = 'bisa dimakan 🍲😋, gasin aja bang👍'
    # Dekode prediksi jika perlu
    return prediction

if st.button("Prediksi"):
    prediction_result = prediction(df)
    st.write('Jamur ini merupkan jamur yang', prediction_result)