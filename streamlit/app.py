import nltk
import json
import pandas as pd

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# Importing the dataset
with open('kampus_merdeka_cmplt.json') as content:
  data1 = json.load(content)

# Mendapatkan semua data ke dalam list
tags = [] # data tag
inputs = [] # data input atau pattern
responses = {} # data respon
words = [] # Data kata
classes = [] # Data Kelas atau Tag
documents = [] # Data Kalimat Dokumen
ignore_words = ['?', '!'] # Mengabaikan tanda spesial karakter

for intent in data1['intents']:
  responses[intent['tag']]=intent['responses']
  for lines in intent['patterns']:
    inputs.append(lines)
    tags.append(intent['tag'])
    for pattern in intent['patterns']:
      w = nltk.word_tokenize(pattern)
      words.extend(w)
      documents.append((w, intent['tag']))
      # add to our classes list
      if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Konversi data json ke dalam dataframe
data = pd.DataFrame({"patterns":inputs, "tags":tags})

label = data['tags'].values.flatten().tolist()
label = list(dict.fromkeys(label))
# print(len(label), "label", label)

from transformers import AutoTokenizer,TFAutoModelForSequenceClassification
import tensorflow as tf
import warnings
from nltk.stem import WordNetLemmatizer
import pandas as pd
import streamlit as st
import time
import random

intents = json.loads(open('kampus_merdeka_cmplt.json').read())

def fetch_data():
    saved_directory = 'model-bert'
    tokenizer = AutoTokenizer.from_pretrained(saved_directory)
    model = TFAutoModelForSequenceClassification.from_pretrained(saved_directory)

    return tokenizer, model

def preprocess(text):
    # define stopword for bahasa
    file_path = 'stopwordbahasa.csv'
    df = pd.read_csv(file_path)

    # Ubah DataFrame menjadi list array menggunakan fungsi values
    stop_words = df.values.flatten().tolist()

    #Lowercase the text
    text = text.lower()

    #Split the text into words
    words = text.split()

    # Perform stemming
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def predict(input_text, intents_json):
    labels = label
    input_text = preprocess(input_text)
    new_X = dict(tokenizer(input_text, padding=True, truncation=True, max_length=50, return_tensors='tf'))
    predictions = model.predict(new_X)
    predictions = labels[tf.argmax(predictions['logits'][0].tolist())]
    print(predictions)

    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == predictions):
            text = random.choice(i['responses'])
            print(text)
            break
        else:
            text = "Maaf chatbot tidak mengerti, coba tanyakan lagi"
    return text

st.markdown("<h1 style='text-align: center;'>Kampus Merdeka Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Ask Any Question About MBKM Program</h1>", unsafe_allow_html=True)


tokenizer, model = fetch_data()

text = ''

st.text("")
st.text("")


with st.form("my_form"):
    input_text = st.text_input('Example: Apa itu kampus merdeka?')
    submitted = st.form_submit_button("Submit")
    

my_bar = st.progress(0)

if submitted:
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)

        text = input_text
        text = predict(input_text, intents)

st.text("")
st.text("Answer")

st.info(text)