import streamlit as st
import re
import string
import pickle
import pandas as pd

# retrieve pickles
model = pickle.load(open('svm_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
selected_features = pickle.load(open('selected_features.pkl', 'rb'))

# Judul aplikasi
st.markdown("<h1 style='text-align: center;'>PREDIKSI SENTIMEN BERITA KEUANGAN MEDIA DI INDONESIA</h1>", unsafe_allow_html=True)

# Preprocessing function
@st.cache_data
def clean(text):
    # Remove newline characters and surrounding whitespace
    clean_text = re.sub(r'\n+', ' ', text)  # Replace newlines with a space
    clean_text = re.sub(r'\s+', ' ', clean_text)  # Normalize whitespace
    clean_text = re.sub(r'\bn\b', ' ', clean_text)  # Normalize whitespace
    clean_text = re.sub(r'\bxd\b', ' ', clean_text)  # Normalize whitespace

    # Remove all uppercase words
    clean_text = re.sub(r'\b[A-Z]+\b', '', clean_text)
    
    # Remove words that end with a period
    clean_text = re.sub(r'\b\w+\.\b', '', clean_text)
    
    # Remove mentions
    clean_text = re.sub(r'@\w+', '', clean_text)
    
    # Convert to lowercase
    clean_text = clean_text.lower()
    
    # Remove domain names
    clean_text = re.sub(r'\b\w*(?:\.com|\.id|\.co)\w*\b', '', clean_text)
    
    # Remove URLs
    clean_text = re.sub(r'https?://\S+|www\.\S+', '', clean_text)
    
    # Remove digits
    clean_text = re.sub(r'\d+', '', clean_text)
    
    # Remove punctuation
    clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
    
    # Remove specific unwanted characters
    clean_text = re.sub(r'[©â€“œ]', '', clean_text)
    
    # Normalize whitespace again
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    # Remove 'com' if it appears as a standalone word
    clean_text = re.sub(r'\bcom\b', '', clean_text)
    
    # Trim trailing whitespace
    clean_text = clean_text.strip()
    
    return clean_text

# Fungsi prediksi
@st.cache_data
def predict(text, _model):
    # Preprocessing teks
    clean_text = clean(text)
    clean_df = pd.DataFrame({
        'Clean Text': [clean_text]
    })
    text_tfidf = tfidf.transform(clean_df["Clean Text"])
    feature_names = tfidf.get_feature_names_out()
    tfidf_matrix = text_tfidf.todense()
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=feature_names)
    tfidf_selected = tfidf_df[selected_features]

    prediction = model.predict(tfidf_selected)

    return prediction[0]

# Mengurangi jarak secara maksimal antara label dan text box
input_text = st.text_area("Masukkan Teks Berita:", height=300)

# CSS untuk merapikan tombol
st.markdown("""
    <style>
        .stButton>button {
            background-color: #74574F;
            color: #FFFFFF;
            border: none;
            border-radius: 5px;
            margin-left: 545px;
            margin-top: -10px;
            padding: 10px 18px;
            font-size: 30px;
            cursor: pointer;
        }
        .result {
            font-family: 'Times New Roman';
            font-size: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Jika tombol ditekan, lakukan prediksi
if st.button("Prediksi Sentimen"):
    if input_text:
        hasil_sentimen = predict(input_text, model)
        label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive', 3: 'Dual'}

        # Menentukan warna berdasarkan hasil prediksi
        if hasil_sentimen == 0:  # Negative
            color = "#9b373d"  # Warna merah
        elif hasil_sentimen == 1:  # Neutral
            color = "#004278"  # Warna biru
        elif hasil_sentimen == 2:  # Positive
            color = "#006c4f"  # Warna hijau
        elif hasil_sentimen == 3:  # Dual
            color = "#4d4e56"  # Warna abu

        # Menampilkan hasil dengan warna yang berbeda
        st.markdown(f"""
        <div style="border-radius: 0px; padding : 10px; background-color: {color}; color: white;" class="result">
            Hasil Prediksi : {label_map.get(hasil_sentimen, 'Unknown')}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.write("Silakan masukkan teks berita untuk diprediksi.")
